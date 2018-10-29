import psycopg2
from . import utils

server_address = "localhost:5432"

def get_database_connection(application_name="bb_behavior"):
    database_host = server_address
    database_port = 5432

    if ":" in database_host:
        database_host, database_port = database_host.split(":")
    return psycopg2.connect("dbname='beesbook' user='reader' host='{}' port='{}' password='reader'".format(database_host, database_port),
                          application_name=application_name)