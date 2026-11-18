import time
import numpy as np 

from importlib import import_module
import carla
import math

from carla_env.core.task_actor.common.task_vehicle import TaskVehicle, TaskVehicle_with_bt


PENALTY_COLLISION_PEDESTRIAN = 0.50
PENALTY_COLLISION_VEHICLE = 0.60
PENALTY_COLLISION_STATIC = 0.65
PENALTY_TRAFFIC_LIGHT = 0.70
PENALTY_STOP = 0.80


class EgoVehicleHandler(object):
    def __init__(self, client, reward_configs, terminal_configs):
        self.ego_vehicle = None
        self.terminal_handlers = {}
        self.reward_handlers = {}
        self.info_buffers = {}
        self.reward_buffers = {}

        self._reward_configs = reward_configs
        self._terminal_configs = terminal_configs
        
         
        self._world = client.get_world()
        self._map = self._world.get_map()
        self._spawn_transforms = self.get_spawn_points(self._map)
            
    @staticmethod
    def get_spawn_points(map):
        all_spawn_points = map.get_spawn_points()
        
        spawn_transforms = []
        for trans in all_spawn_points:
            wp = map.get_waypoint(trans.location)
            
            if wp.is_junction: # junction is a intersection.
                wp_prev = wp 
                while wp_prev.is_junction:
                    wp_prev = wp_prev.previous(1.0)[0]
                spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
                if map.name == 'Town03' and (wp_prev.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
            
            else:
                spawn_transforms.append([wp.road_id, wp.transform])
                if map.name == 'Town03' and (wp_prev.road_id == 44):
                    for _ in range(100):
                        spawn_transforms.append([wp_prev.road_id, wp_prev.transform])
            
        return spawn_transforms
    
    def reset(self, task_config):
        actor_config = task_config['actors']['hero']
        route_config = task_config['routes']['hero']
        endless_config = task_config.get('endless')
        
        if endless_config is not None:
            endless_config = endless_config['hero']

        ev_spawn_locations = []

        
        bp_filter = actor_config['model']
        # print(f"the aviliable blueprint list: {[bp.id for bp in self._world.get_blueprint_library()]}")
        blueprint = np.random.choice(self._world.get_blueprint_library().filter(bp_filter))

        if len(route_config) == 0: # this means if we are not using predefined routes! And therefore we use one of the available spawn_points.
            spawn_transform = np.random.choice([x[1] for x in self._spawn_transforms])
        else:
            spawn_transform = route_config[0] # first waypoint of the route.

        
        wp = self._map.get_waypoint(spawn_transform.location)

        # Adjust the z-coordinate of the spawn location to ensure the vehicle spawns slightly above the ground.
        # This helps to avoid potential collisions or embedding issues with the ground surface.
        spawn_transform.location.z = wp.transform.location.z + 1.321

        carla_vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)


        self._world.tick()

        if endless_config is None:
            endless = False
        else:
            endless = endless_config
            
        target_transforms = route_config[1:]
        
        self.ego_vehicle = TaskVehicle(carla_vehicle, target_transforms, self._spawn_transforms, endless)

        
        self.reward_handlers = self._build_instance(self._reward_configs, self.ego_vehicle)

        self.terminal_handlers = self._build_instance(self._terminal_configs, self.ego_vehicle)

        self.reward_buffers = []
        self.info_buffers = {
            'collisions_layout': [],
            'collisions_vehicle': [],
            'collisions_pedestrian': [],
            'collisions_others': [],
            'red_light': [],
            'encounter_light': [],
            'stop_infraction': [],
            'encounter_stop': [],
            'route_dev': [],
            'vehicle_blocked': [],
            'outside_lane': [],
            'wrong_lane': []
        }

        ev_spawn_locations.append(carla_vehicle.get_location())
        return ev_spawn_locations


    @staticmethod
    def _build_instance(config, ego_vehicle):
        module_str, class_str = config['entry_point'].split(':')
        _Class = getattr(import_module('carla_env.core.task_actor.ego_vehicle.'+module_str), class_str)
        return _Class(ego_vehicle, **config.get('kwargs', {}))

    def apply_control(self, control):
        self.ego_vehicle.vehicle.apply_control(control)

    def tick(self, timestamp):
        reward_dict, done_dict, info_dict = {}, {}, {}

        info_criteria = self.ego_vehicle.tick(timestamp)
        info = info_criteria.copy()

        done, timeout, terminal_reward, terminal_debug = self.terminal_handlers.get(timestamp)
        reward, reward_debug = self.reward_handlers.get(terminal_reward)

        reward_dict = reward
        done_dict = done
        info_dict = info
        info_dict['timeout'] = timeout
        info_dict['reward_debug'] = reward_debug
        info_dict['terminal_debug'] = terminal_debug

        # accumulate into buffers
        self.reward_buffers.append(reward)

        if info['collision']:
            if info['collision']['collision_type'] == 0:
                self.info_buffers['collisions_layout'].append(info['collision'])
            elif info['collision']['collision_type'] == 1:
                self.info_buffers['collisions_vehicle'].append(info['collision'])
            elif info['collision']['collision_type'] == 2:
                self.info_buffers['collisions_pedestrian'].append(info['collision'])
            else:
                self.info_buffers['collisions_others'].append(info['collision'])
        if info['run_red_light']:
            self.info_buffers['red_light'].append(info['run_red_light'])
        if info['encounter_light']:
            self.info_buffers['encounter_light'].append(info['encounter_light'])
        if info['run_stop_sign']:
            if info['run_stop_sign']['event'] == 'encounter':
                self.info_buffers['encounter_stop'].append(info['run_stop_sign'])
            elif info['run_stop_sign']['event'] == 'run':
                self.info_buffers['stop_infraction'].append(info['run_stop_sign'])
        if info['route_deviation']:
            self.info_buffers['route_dev'].append(info['route_deviation'])
        if info['blocked']:
            self.info_buffers['vehicle_blocked'].append(info['blocked'])
        if info['outside_route_lane']:
            if info['outside_route_lane']['outside_lane']:
                self.info_buffers['outside_lane'].append(info['outside_route_lane'])
            if info['outside_route_lane']['wrong_lane']:
                self.info_buffers['wrong_lane'].append(info['outside_route_lane'])
        # save episode summary
        if done:
            info_dict['episode_event'] = self.info_buffers
            info_dict['episode_event']['timeout'] = info['timeout']
            info_dict['episode_event']['route_completion'] = info['route_completion']

            total_length = float(info['route_completion']['route_length_in_m']) / 1000
            completed_length = float(info['route_completion']['route_completed_in_m']) / 1000
            total_length = max(total_length, 0.001)
            completed_length = max(completed_length, 0.001)

            outside_lane_length = np.sum([x['distance_traveled']
                                            for x in self.info_buffers['outside_lane']]) / 1000
            wrong_lane_length = np.sum([x['distance_traveled']
                                        for x in self.info_buffers['wrong_lane']]) / 1000

            if self.ego_vehicle._endless:
                score_route = completed_length
            else:
                if info['route_completion']['is_route_completed']:
                    score_route = 1.0
                else:
                    score_route = completed_length / total_length

            n_collisions_layout = int(len(self.info_buffers['collisions_layout']))
            n_collisions_vehicle = int(len(self.info_buffers['collisions_vehicle']))
            n_collisions_pedestrian = int(len(self.info_buffers['collisions_pedestrian']))
            n_collisions_others = int(len(self.info_buffers['collisions_others']))
            n_red_light = int(len(self.info_buffers['red_light']))
            n_encounter_light = int(len(self.info_buffers['encounter_light']))
            n_stop_infraction = int(len(self.info_buffers['stop_infraction']))
            n_encounter_stop = int(len(self.info_buffers['encounter_stop']))
            n_collisions = n_collisions_layout + n_collisions_vehicle + n_collisions_pedestrian + n_collisions_others

            score_penalty = 1.0 * (1 - (outside_lane_length+wrong_lane_length)/completed_length) \
                * (PENALTY_COLLISION_STATIC ** n_collisions_layout) \
                * (PENALTY_COLLISION_VEHICLE ** n_collisions_vehicle) \
                * (PENALTY_COLLISION_PEDESTRIAN ** n_collisions_pedestrian) \
                * (PENALTY_TRAFFIC_LIGHT ** n_red_light) \
                * (PENALTY_STOP ** n_stop_infraction) \

            if info['route_completion']['is_route_completed'] and n_collisions == 0:
                is_route_completed_nocrash = 1.0
            else:
                is_route_completed_nocrash = 0.0

            info_dict['episode_stat'] = {
                'score_route': score_route,
                'score_penalty': score_penalty,
                'score_composed': max(score_route*score_penalty, 0.0),
                'length': len(self.reward_buffers),
                'reward': np.sum(self.reward_buffers),
                'timeout': float(info['timeout']),
                'is_route_completed': float(info['route_completion']['is_route_completed']),
                'is_route_completed_nocrash': is_route_completed_nocrash,
                'route_completed_in_km': completed_length,
                'route_length_in_km': total_length,
                'percentage_outside_lane': outside_lane_length / completed_length,
                'percentage_wrong_lane': wrong_lane_length / completed_length,
                'collisions_layout': n_collisions_layout / completed_length,
                'collisions_vehicle': n_collisions_vehicle / completed_length,
                'collisions_pedestrian': n_collisions_pedestrian / completed_length,
                'collisions_others': n_collisions_others / completed_length,
                'red_light': n_red_light / completed_length,
                'light_passed': n_encounter_light-n_red_light,
                'encounter_light': n_encounter_light,
                'stop_infraction': n_stop_infraction / completed_length,
                'stop_passed': n_encounter_stop-n_stop_infraction,
                'encounter_stop': n_encounter_stop,
                'route_dev': len(self.info_buffers['route_dev']) / completed_length,
                'vehicle_blocked': len(self.info_buffers['vehicle_blocked']) / completed_length
            }

        #done_dict['__all__'] = all(done for obs_id, done in done_dict.items())
        return reward_dict, done_dict, info_dict

    def clean(self):
        if self.ego_vehicle is not None:
            self.ego_vehicle.clean()
        self.reward_handlers = {}
        self.terminal_handlers = {}
        self.info_buffers = {}
        self.reward_buffers = {}


class QtruckHander(EgoVehicleHandler):
    def __init__(self, client, reward_configs, terminal_configs):
        super(QtruckHander, self).__init__(client, reward_configs, terminal_configs)
        self.ego_vehicle_bt = None

    def reset(self, task_config):
        actor_config = task_config['actors']['hero']
        route_config = task_config['routes']['hero']
        endless_config = task_config.get('endless')
        
        if endless_config is not None:
            endless_config = endless_config['hero']

        ev_spawn_locations = []

        
        # bp_filter = actor_config['model']
        # # print(f"the aviliable blueprint list: {[bp.id for bp in self._world.get_blueprint_library()]}")
        # blueprint = np.random.choice(self._world.get_blueprint_library().filter(bp_filter))
        # blueprint_bt = self._world.get_blueprint_library().filter('qtruckbt')[0]
        # print("blueprint_bt in QtruckHandler: ", blueprint_bt.id)

        # if len(route_config) == 0: # this means if we are not using predefined routes! And therefore we use one of the available spawn_points.
        #     spawn_transform = np.random.choice([x[1] for x in self._spawn_transforms])
        # else:
        #     spawn_transform = route_config[0] # first waypoint of the route.

        
        # wp = self._map.get_waypoint(spawn_transform.location)

        # # Adjust the z-coordinate of the spawn location to ensure the vehicle spawns slightly above the ground.
        # # This helps to avoid potential collisions or embedding issues with the ground surface.
        # spawn_transform.location.z = wp.transform.location.z +1
        # # 得到当前车头的yaw角度
        # yaw = spawn_transform.rotation.yaw
        # # 定义车头与拖挂的偏移
        # distance = 5
        # angle_rad = math.radians(yaw)
        # _x = distance * math.cos(angle_rad)
        # _y = distance * math.sin(angle_rad)
        # spawn_transform_bt = carla.Transform(
        #     carla.Location(x=spawn_transform.location.x - _x, 
        #                 y=spawn_transform.location.y - _y, 
        #                 z=spawn_transform.location.z), 
        #     carla.Rotation(pitch=spawn_transform.rotation.pitch,
        #                     yaw=spawn_transform.rotation.yaw,
        #                     roll=spawn_transform.rotation.roll)
        # )

        # carla_vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)
        # print("spawn_transform_bt in QtruckHandler: ", spawn_transform_bt)
        # carla_vehicle_bt = self._world.spawn_actor(blueprint_bt, spawn_transform_bt)

        carla_vehicle, carla_vehicle_bt = self.spawn_vehicle_with_bt(task_config)
        
        while True:
            if (carla_vehicle_bt is None) or (carla_vehicle is None):
                carla_vehicle, carla_vehicle_bt = self.spawn_vehicle_with_bt(task_config)
            else:
                break

        vehicle_control = carla.VehicleControl()
        vehicle_control.brake = 0.0000000001
        carla_vehicle_bt.apply_control(vehicle_control)
        
        # self._world.tick()
        while True:
            try:
                self._world.tick()
                break
            except Exception as e:
                print(f"tick 丢失，重新发送: {e}")
                time.sleep(2)


        if endless_config is None:
            endless = False
        else:
            endless = endless_config
            
        target_transforms = route_config[1:]
        
        # self.ego_vehicle = TaskVehicle(carla_vehicle, target_transforms, self._spawn_transforms, endless)
        # 相比于TaskVehicle，TaskVehicle_with_bt增加了bt的碰撞检测，并将碰撞信息转换到车头坐标系下。目前还未进行坐标变换
        
        print("carla_vehicle_bt id in QtruckHandler: ", carla_vehicle_bt.id)
        

        ego_vehicle_with_bt = TaskVehicle_with_bt(carla_vehicle, carla_vehicle_bt, target_transforms, self._spawn_transforms, endless)

        self.ego_vehicle = ego_vehicle_with_bt

        
        # TODO: reward and terminal handlers for vehicle with bt!!!
        self.reward_handlers = self._build_instance(self._reward_configs, self.ego_vehicle)

        self.terminal_handlers = self._build_instance(self._terminal_configs, self.ego_vehicle)

        self.reward_buffers = []
        self.info_buffers = {
            'collisions_layout': [],
            'collisions_vehicle': [],
            'collisions_pedestrian': [],
            'collisions_others': [],
            'red_light': [],
            'encounter_light': [],
            'stop_infraction': [],
            'encounter_stop': [],
            'route_dev': [],
            'vehicle_blocked': [],
            'outside_lane': [],
            'wrong_lane': []
        }

        ev_spawn_locations.append(carla_vehicle.get_location())
        return ev_spawn_locations
    
    def spawn_vehicle_with_bt(self, task_config):
        route_config = task_config['routes']['hero']
        actor_config = task_config['actors']['hero']

        bp_filter = actor_config['model']
        blueprint = np.random.choice(self._world.get_blueprint_library().filter(bp_filter))
        blueprint_bt = self._world.get_blueprint_library().filter('qtruckbt')[0]

        if len(route_config) == 0:
            spawn_transform = np.random.choice([x[1] for x in self._spawn_transforms])
        else:
            spawn_transform = route_config[0]

        wp = self._map.get_waypoint(spawn_transform.location)
        spawn_transform.location.z = wp.transform.location.z + 1.321
        yaw = spawn_transform.rotation.yaw
        distance = 5
        angle_rad = math.radians(yaw)
        _x = distance * math.cos(angle_rad)
        _y = distance * math.sin(angle_rad)
        spawn_transform_bt = carla.Transform(
            carla.Location(x=spawn_transform.location.x - _x, 
                        y=spawn_transform.location.y - _y, 
                        z=spawn_transform.location.z), 
            carla.Rotation(pitch=spawn_transform.rotation.pitch,
                            yaw=spawn_transform.rotation.yaw,
                            roll=spawn_transform.rotation.roll)
        )
        carla_vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)
        carla_vehicle_bt = self._world.try_spawn_actor(blueprint_bt, spawn_transform_bt)
        
        
            
        return carla_vehicle, carla_vehicle_bt

    

    
