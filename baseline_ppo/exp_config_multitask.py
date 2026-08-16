"""Multi-task unsignalized intersection example.

Step 4 (real env): one RL ego vehicle (AEV) and background IDM traffic on
a 4-way, single-lane, non-signalized intersection, trained against the
paper-faithful env (61-dim state incl. task-intent mu_a, 3-discrete
action, flat 3-case reward). See
PROJECTS/1_MultiTask_Intersection/Step3_docs/rl_formulation.md for the
full derivation.

Network note: the template originally used for the Step 4 network smoke
test (500m_Unregulated_Junction.net.xml) has 500 m approach arms -- at the
paper's 8 m/s speed cap that's ~62 s just to cross one arm, incompatible
with the paper's 25 s episode. Switched to 50mIntersection.net.xml (same
edge-naming convention, same single-lane topology, ~22.8 m arms, non-
signalized "priority"-type junction), where a full straight crossing
takes ~6 s.

Timing note: sim_step=1/15 and sims_per_step=15 together mean each RL
(policy) step advances the simulation by exactly 1 second, matching the
paper's 15 Hz simulation / 1 Hz decision frequency. horizon=25 then
matches the paper's 25 s episode duration exactly.
"""
import os

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams, InFlows
from flow.controllers import RLController, IDMController
from flow.envs.multitask_intersection import MultiTaskIntersectionEnv
from flow.networks.all_turning_intersection import AllTurningIntersectionNetwork

# number of decision (policy) steps per episode -- paper: 25 s @ 1 Hz
HORIZON = 25
# number of rollouts (episodes) per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 10

# template network file: 4-way, single-lane, non-signalized intersection.
# Edge ids (E#L-X, E#D-X, E#R-X, E#T-X in; E#X-T, E#X-D, E#X-L, E#X-R out)
# match AllTurningIntersectionNetwork.specify_routes() exactly.
NET_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'exp_inputs', 'scenarios',
    '50mIntersection.net.xml')

vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=1,
        max_speed=8,
    ),
    num_vehicles=0)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=1,
        max_speed=8,
    ),
    num_vehicles=0)

inflow = InFlows()
for edge in ["E#L-X", "E#D-X", "E#R-X", "E#T-X"]:
    inflow.add(
        veh_type="human",
        edge=edge,
        probability=0.06,
        depart_lane="free",
        depart_speed="random",
        begin=1)
# single RL (ego/AEV) vehicle, spawned once near the start of the episode.
# NOTE: deliberately not using `number=1` here. AllTurningIntersectionNetwork
# gives edge "E#L-X" 3 stochastic sub-routes, so Flow splits this one
# inflow.add() call into 3 separate <flow> entries in the generated
# .rou.xml, one per sub-route, each getting `number` scaled by that
# sub-route's probability (e.g. int(1 * 0.35) = 0) -- with number=1 every
# sub-flow rounds down to number="0" and the ego never spawns. Using a
# narrow 1-second departure window (begin=1, end=2) with probability=1.0
# instead guarantees exactly one spawn attempt without hitting that
# rounding-to-zero bug.
inflow.add(
    veh_type="rl",
    edge="E#L-X",
    probability=1.0,
    depart_lane="free",
    depart_speed="random",
    begin=1,
    end=2)

flow_params = dict(
    # name of the experiment
    exp_tag="multitask_intersection",

    # name of the flow environment the experiment is running on
    env_name=MultiTaskIntersectionEnv,

    # name of the network class the experiment is running on
    network=AllTurningIntersectionNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=1 / 15,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=0,
        sims_per_step=15,
        clip_actions=False,
        additional_params={
            "num_surrounding": 9,
            "detection_radius": 48,
            "target_velocity": 8,
            "speed_gain": 5,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams) --
    # geometry comes from the template net file, not additional_params
    net=NetParams(
        inflows=inflow,
        template=NET_FILE,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams) -- all vehicles enter via inflow
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
