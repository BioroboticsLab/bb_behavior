import psycopg2

server_address = "localhost:5432"
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
    return psycopg2.connect("dbname='beesbook' user='{}' host='{}' port='{}' password='{}'".format(user, database_host, database_port, password),
                          application_name=application_name)