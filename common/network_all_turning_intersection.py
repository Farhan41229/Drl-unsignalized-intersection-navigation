from flow.networks import Network
# some mathematical operations that may be used
from numpy import pi, sin, cos, linspace

#
import xml.etree.ElementTree as ElementTree
from lxml import etree

#
# default sumo probability value  TODO (ak): remove
DEFAULT_PROBABILITY = 0
# default sumo vehicle length value (in meters) TODO (ak): remove
DEFAULT_LENGTH = 5
# default sumo vehicle class class TODO (ak): remove
DEFAULT_VCLASS = 0


ADDITIONAL_NET_PARAMS={
    "length": 30,
    "num_lanes": 1,
    "speed_limit": 30,
}


class AllTurningIntersectionNetwork(Network):

    """
    E#R-X means Edge # strating from Right side edge to Itersection named 'X'
    E#X-R means Edge # strating from Itersection named 'X to Right side edge
    L=Left side edge
    T=Top side edge
    D=Down side edge
    """

    #Routes with Stotastics
    def specify_routes(self, net_params):
        rts = {

            "E#T-X": [
                (["E#T-X", "E#X-D"], 0.5),  # 50% Straight Move
                (["E#T-X", "E#X-L"], 0.25), # 25% Right Move
                (["E#T-X", "E#X-R"], 0.25),  # 25% Left Move
            ],
            "E#D-X": [
                (["E#D-X", "E#X-T"], 0.5),  # 50% Straight Move
                (["E#D-X", "E#X-R"], 0.25), # 25% Right Move
                (["E#D-X", "E#X-L"], 0.25), # 25% Left Move
            ],
            "E#L-X": [
                (["E#L-X", "E#X-R"], 0.5),  # 50% Straight Move
                (["E#L-X", "E#X-D"], 0.25),  # 25% Right Move
                (["E#L-X", "E#X-T"], 0.25),  # 25% Left Move
            ],

            "E#R-X": [
                (["E#R-X", "E#X-L"], 0.1),  # 50% Straight Move
                (["E#R-X", "E#X-T"], 0.8),  # 25% Right Move
                (["E#R-X", "E#X-D"], 0.1),  # 25% Left Move
            ],
        }

        return rts


    def specify_routes(self, net_params):

        # config_1_non_colliding: # 2 Simultanious Left Turn-(from Top & Down Edge) + 2 Simultanious Right Turn-(from Left & Right Edge)
        rts_config_1_non_colliding = {

            #2 Simultanious Left Turn-(from Top & Down Edge)
           "E#T-X": ["E#T-X", "E#X-R"],
           "E#D-X": ["E#D-X", "E#X-L"],

           #2 Simultanious Right Turn-(from Left & Right Edge)
           "E#L-X": ["E#L-X", "E#X-D"],
           "E#R-X": ["E#R-X", "E#X-T"]
           }

        # config_2_non_colliding # 2 Simultanious Left Turn-(from Left & Right Edge) + 2 Simultanious Right Turn-(from Top & Down Edge)
        rts_config_2_non_colliding = {
           "E#L-X": ["E#L-X", "E#X-T"],
           "E#R-X": ["E#R-X", "E#X-D"],

           "E#T-X": ["E#T-X", "E#X-L"],
           "E#D-X": ["E#D-X", "E#X-R"]
           }

        # config_3: nonColliding_top_down # 2 Simultanious Straight Move-(from Top & Down Edge) + 2 Simultanious Right Turn-(from Top & Down Edge) # Colliding from Left & Right Edge
        rts_config_3_nonColliding_top_down_colliding_left_right = {
           "E#T-X": [
                        (["E#T-X", "E#X-D"], 0.5),  # 50% probability to follow this route
                        (["E#T-X", "E#X-L"], 0.5),  # 50% probability to follow this route
                    ],
           "E#D-X": [
                        (["E#D-X", "E#X-T"], 0.5),  # 50% probability to follow this route
                        (["E#D-X", "E#X-R"], 0.5),  # 50% probability to follow this route
                    ],

           "E#L-X": ["E#L-X", "E#X-R"], # May Collide with Upper 2
           "E#R-X": ["E#R-X", "E#X-L"], # May Collide with Upper 2
           }

        # config_4: Non-Colliding# 2 Simultanious Straight Move-(from Left & Right Edge) + 2 Simultanious Right Turn-(from Left & Right Edge) # Colliding from Top & Down Edge
        rts_config_4_nonColliding_left_right_colliding_top_down = {
           "E#L-X": [
                        (["E#L-X", "E#X-R"], 0.5),  # 50% probability to follow this route
                        (["E#L-X", "E#X-D"], 0.5),  # 50% probability to follow this route
                    ],
           "E#R-X": [
                        (["E#R-X", "E#X-L"], 0.5),  # 50% probability to follow this route
                        (["E#R-X", "E#X-T"], 0.5),  # 50% probability to follow this route
                    ],

           "E#T-X": ["E#T-X", "E#X-D"], # May Collide with Upper 2
           "E#D-X": ["E#D-X", "E#X-T"], # May Collide with Upper 2
           }

        # config_5: #Routes with Stotastics # all (3) possible moves -(from Left & Right & Top & Down Edge)
        rts_config_5_stocastic = {

            "E#T-X": [
                (["E#T-X", "E#X-D"], 0.5),  # 50% Straight Move
                (["E#T-X", "E#X-L"], 0.25), # 25% Right Move
                (["E#T-X", "E#X-R"], 0.5),  # 25% Left Move
            ],
            "E#D-X": [
                (["E#D-X", "E#X-T"], 0.5),  # 50% Straight Move
                (["E#D-X", "E#X-R"], 0.25), # 25% Right Move
                (["E#D-X", "E#X-L"], 0.25), # 25% Left Move
            ],
            "E#L-X": [
                (["E#L-X", "E#X-R"], 0.5),  # 50% Straight Move
                (["E#L-X", "E#X-D"], 0.25),  # 25% Right Move
                (["E#L-X", "E#X-T"], 0.25),  # 25% Left Move
            ],

            "E#R-X": [
                (["E#R-X", "E#X-L"], 0.5),  # 50% Straight Move
                (["E#R-X", "E#X-T"], 0.25),  # 25% Right Move
                (["E#R-X", "E#X-D"], 0.25),  # 25% Left Move
            ],
        }

        # config_6_all_stright_move: Colliding#(from Left, Right, Top, Down  Edge )
        rts_config_6_all_stright_move = {
           "E#T-X": ["E#T-X", "E#X-D"],
           "E#D-X": ["E#D-X", "E#X-T"],

           "E#L-X": ["E#L-X", "E#X-R"],
           "E#R-X": ["E#R-X", "E#X-L"]
           }

        # config_7_all_right_turn_move: Non-Colliding#(from Left, Right, Top, Down  Edge )
        rts_config_7_all_right_turn_move = {
           "E#T-X": ["E#T-X", "E#X-L"],
           "E#D-X": ["E#D-X", "E#X-R"],

           "E#L-X": ["E#L-X", "E#X-D"],
           "E#R-X": ["E#R-X", "E#X-T"]
           }

        # config_8_custom_DEFINE BY YOURSELF
        rts_config_8_custom = {
           "E#T-X": ["E#T-X", "E#X-L"],
           "E#D-X": ["E#D-X", "E#X-R"],

           "E#L-X": ["E#L-X", "E#X-D"],
           "E#R-X": ["E#R-X", "E#X-D"]
           }

        rts_random = {

            "E#T-X": [
                (["E#T-X", "E#X-D"], 0.35),  # 50% Straight Move
                (["E#T-X", "E#X-L"], 0.35), # 25% Right Move
                (["E#T-X", "E#X-R"], 0.30),  # 25% Left Move
            ],
            "E#D-X": [
                (["E#D-X", "E#X-T"], 0.30),  # 50% Straight Move
                (["E#D-X", "E#X-R"], 0.35), # 25% Right Move
                (["E#D-X", "E#X-L"], 0.35), # 25% Left Move
            ],
            "E#L-X": [
                (["E#L-X", "E#X-R"], 0.30),  # 50% Straight Move
                (["E#L-X", "E#X-D"], 0.35),  # 25% Right Move
                (["E#L-X", "E#X-T"], 0.35),  # 25% Left Move
            ],

            "E#R-X": [
                (["E#R-X", "E#X-L"], 0.35),  # 50% Straight Move
                (["E#R-X", "E#X-T"], 0.35),  # 25% Right Move
                (["E#R-X", "E#X-D"], 0.30),  # 25% Left Move
            ],
        }

        rts = rts_random
        return rts



    def _vehicle_type_custom(filename):
        """Import vehicle type data from a *.add.xml file.

        This is a utility function for outputting all the type of vehicle.

        Parameters
        ----------
        filename : str
            path to the vtypes.add.xml file to load

        Returns
        -------
        dict or None
            the key is the vehicle_type id and the value is a dict we've type
            of the vehicle, depart edges, depart Speed, departPos. If no
            filename is provided, this method returns None as well.
        """
        if filename is None:
            return None

        parser = etree.XMLParser(recover=True)
        tree = ElementTree.parse(filename, parser=parser)

        root = tree.getroot()
        veh_type = {}

        # this hack is meant to support the LuST network and Flow networks
        root = [root] if len(root.findall('vTypeDistribution')) == 0 \
            else root.findall('vTypeDistribution')

        for r in root:
            for vtype in r.findall('vType'):
                # TODO: make for everything
                veh_type[vtype.attrib['id']] = {
                    'vClass': vtype.attrib.get('vClass', DEFAULT_VCLASS),
                    'accel': vtype.attrib['accel'],
                    'decel': vtype.attrib['decel'],
                    'sigma': vtype.attrib['sigma'],
                    'length': vtype.attrib.get('length', DEFAULT_LENGTH),
                    'minGap': vtype.attrib['minGap'],
                    'maxSpeed': vtype.attrib['maxSpeed'],
                    'probability': vtype.attrib.get(
                        'probability', DEFAULT_PROBABILITY),
                    'speedDev': vtype.attrib['speedDev']
                }

        return veh_type
