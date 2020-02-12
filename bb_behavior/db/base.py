import psycopg2

server_address = "localhost:5432"
database_name = "beesbook"
database_user = "reader"
database_password = "reader"

def get_database_connection(application_name="bb_behavior", user=None, password=None):
    database_host = server_address
    database_port = 5432
    if user is None:
        user = database_user
    if password is None:
        password = database_password
    if ":" in database_host:
        database_host, database_port = database_host.split(":")
    return psycopg2.connect("dbname='{}' user='{}' host='{}' port='{}' password='{}'".format(database_name, user, database_host, database_port, password),
                          application_name=application_name)


"""
    Different seasons can be placed in the same database. The tables will have different names then.
    The following configuration can be used to change the data sources used by the helper tools.
"""

SEASON_BERLIN_2016 = dict(
    identifier="berlin_2016",
    bb_detections = "bb_detections_2016_stitched",
    bb_alive_bees = "alive_bees_2016",
    bb_frame_metadata = "bb_frame_metadata_2016",
    bb_framecontainer_metadata = "framecontainer_metadata_2016",
    temp_tablespace = "ssdspace"
)

SEASON_KONSTANZ_2018 = dict(
    identifier="konstanz_2018",
    bb_detections = "bb_detections_2018_konstanz",
    bb_frame_metadata = "bb_frame_metadata_2018_konstanz"
)

beesbook_season_config = SEASON_BERLIN_2016.copy()

def set_season_berlin_2016():
    global beesbook_season_config
    beesbook_season_config = SEASON_BERLIN_2016.copy()
def set_season_konstanz_2018():
    global beesbook_season_config
    beesbook_season_config = SEASON_KONSTANZ_2018.copy()

def get_season_identifier():
    return beesbook_season_config["identifier"]
def get_detections_tablename():
    return beesbook_season_config["bb_detections"]
def get_alive_bees_tablename():
    return beesbook_season_config["bb_alive_bees"]
def get_frame_metadata_tablename():
    return beesbook_season_config["bb_frame_metadata"]
def get_framecontainer_metadata_tablename():
    return beesbook_season_config["bb_framecontainer_metadata"]
def get_temp_tablespace():
    if "temp_tablespace" in beesbook_season_config:
        return beesbook_season_config["temp_tablespace"]
    return None