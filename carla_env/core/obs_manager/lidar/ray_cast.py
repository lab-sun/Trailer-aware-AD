import numpy as np
import copy
import weakref
import carla
from queue import Queue, Empty
from gym import spaces
from math import inf
from scipy.spatial.transform import Rotation as R

from carla_env.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    """
    Template configs:
    obs_configs = {
        module: lidar.ray_cast
        location: [3, -1.275, 1.15]
        rotation: [0.0, -45, 0]
        range: 50.0
        rotation_frequency: 10
        points_per_second: 640000
        horizontal_fov: 360 # deg
        upper_fov: 15 # deg
        lower_fov: -16 # deg
        channel: 32 # int
    }
    frame_stack: [Point(t-2), Point(t-1), Point(t)]
    """

    def __init__(self, obs_configs):

        self._sensor_type = 'lidar.ray_cast'

        self._range = obs_configs['range']
        self._channels = obs_configs['channels']
        self._points_per_second = obs_configs['points_per_second']
        self._upper_fov = obs_configs['upper_fov']
        self._lower_fov = obs_configs['lower_fov']
        self._horizontal_fov = obs_configs['horizontal_fov']
        self._rotation_frequency = obs_configs['rotation_frequency']

        location = carla.Location(
            x=float(obs_configs['location'][0]),
            y=float(obs_configs['location'][1]),
            z=float(obs_configs['location'][2]))
        rotation = carla.Rotation(
            roll=float(obs_configs['rotation'][0]),
            pitch=float(obs_configs['rotation'][1]),
            yaw=float(obs_configs['rotation'][2]))

        self._lidar_transform = carla.Transform(location, rotation)

        self._sensor = None
        self._queue_timeout = 10.0
        self._points_queue = None

        super(ObsManager, self).__init__()

    # TODO: define obs space for pointcloud data!! DONE
    # data shape 需要确认点个数的定义，现在的定义方式是 每秒点数 / 仿真器频率
    def _define_obs_space(self): 

        carla_fps = 10 # Hz carla step的频率，每0.1s更新一次场景，获得一次数据
        self.obs_space = spaces.Dict({
            'frame': spaces.Discrete(2**32-1),
            'data': spaces.Box(
                low=-inf, high=inf, shape=(int(self._points_per_second / carla_fps), 3), dtype=np.float32)
        })

    def attach_ego_vehicle(self, parent_actor):
        self._points_queue = Queue()

        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find(f"sensor.{self._sensor_type}")
        bp.set_attribute('range', str(self._range))
        bp.set_attribute('rotation_frequency', str(self._rotation_frequency))
        bp.set_attribute('points_per_second', str(self._points_per_second))
        bp.set_attribute('horizontal_fov', str(self._horizontal_fov))
        bp.set_attribute('upper_fov', str(self._upper_fov))
        bp.set_attribute('lower_fov', str(self._lower_fov))
        bp.set_attribute('channels', str(self._channels))
        
        # if self._distort_images:
        #     # set in leaderboard
        #     bp.set_attribute('lens_circle_multiplier', str(3.0))
        #     bp.set_attribute('lens_circle_falloff', str(3.0))
        #     bp.set_attribute('chromatic_aberration_intensity', str(0.5))
        #     bp.set_attribute('chromatic_aberration_offset', str(0))

        self._sensor = self._world.spawn_actor(bp, self._lidar_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda points: self._parse_points(weak_self, points))

    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._points_queue.qsize() <= 1
        # frame, data = self._points_queue.get(True, self._queue_timeout)
        try: 
            frame, data = self._points_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('LIDAR sensor took too long!')

        obs = {'frame': frame,
               'data': data}

        return obs

    def clean(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        self._sensor = None
        self._world = None

        self._points_queue = None

    @staticmethod  # preprocess points?
    def _parse_points(weak_self, carla_points):
        self = weak_self()

        np_pc = np.frombuffer(carla_points.raw_data, dtype=np.dtype("float32")) # dtype float32?
        np_pc = copy.deepcopy(np_pc)
        
        np_pc = np.reshape(np_pc, (-1, 4)) # 此处假设 点云数据有4个通道，x, y, z, intensity；如果没有intensity，应修改此处
        np_pc = np_pc[(np_pc[:, 0] <= 0) & (np_pc[:, 0] >= -15) & (np_pc[:, 1] >= -5) &  (np_pc[:, 1] <= 5) & (np_pc[:, 2] >= 0) & (np_pc[:, 2] <= 4)] # filter points based on x, y, z range
        num_points = 2000
        # Downsample to 2000 points
        if np_pc.shape[0] > num_points:
            indices = np.random.choice(np_pc.shape[0], num_points, replace=False)
            np_pc = np_pc[indices]
        # print(f"pc after filter: {np_pc}")
        # print(f"pc shape after filter: {np_pc.shape}")
        np_pc_ = np_pc[:, :3]
        # Filter out points where x or y is less than 0.5
        
        # print(f"pc before transform: {np_pc}")

        # 转化到 qtruck 坐标系
        # Transform points to the vehicle's coordinate system
        lidar_transform = self._lidar_transform
        lidar_to_vehicle = np.array(lidar_transform.get_inverse_matrix())
        # lidar_to_vehicle = np.array(lidar_transform)

        # 获取LiDAR相对于车辆的平移和旋转
        location = lidar_transform.location
        # print(f"location: {location}")
        rotation = lidar_transform.rotation
        # print(f"rotation: {rotation}")

        # Homogeneous coordinates for point cloud
        homogeneous_pc = np.hstack((np_pc_, np.ones((np_pc_.shape[0], 1), dtype=np.float32)))

        # Apply transformation
        transformed_pc = np.dot(lidar_to_vehicle, homogeneous_pc.T).T
        # print(f"pc after transform: {transformed_pc}")
        

        # 把 instensity 添加回来
        np_pc__ = np.hstack((transformed_pc[:, :3], np_pc[:, 3:4]))

        self._points_queue.put((carla_points.frame, np_pc__))
