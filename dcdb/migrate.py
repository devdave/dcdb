""""
    Migration command line interface

    setup
        create a migration repository
        create a migrate_cli.py which acts as the repository tool as well as hold its settings
        Create a Migration table for tracking database state

    checkdb
        if config.module is set and returns an object (module/package/anything with __dict__) then
            it compares the provided models against the database and lists differences

    status
        prints the Migration table
        lists unrun stages
        lists run stages

    create
        --with-check - if present will create a new migration from the contents of
        checkdb.

        creates ###_name__{Hash(###+name).py
        creates ###_name_dir_{Hash(###+Name)/
            ___init__.py

    push
        --latest - runs only the latest stage and not any subsequent stages
        --dry - bool if True does a dry run


"""
import click
import sqlite3
import pathlib
import dcdb


template = """
from dcdb import migrate

config = dict()
config['SRC'] = "{database_filepath}"
config['REPO'] = "{changeset_dirpath}"
config['models'] = "path.2.module"

if __name__ == "__main__":
    migrate.run(config)
"""


@click.group()
def cli():
    pass

@click.command()
@click.argument("filepath")
def checkdb(filepath):
    filepath = pathlib.Path(filepath)
    assert filepath.exists(), f"Unable to find {filepath}"
    conn = dcdb.DBConnection(filepath)
    records = conn.execute("SELECT type,name, tbl_name FROM SQLITE_MASTER where type = 'table'")
    row = records.fetchone()
    spacer = 30
    print("\t".join([k.ljust(spacer) for k in row.keys()]))
    print("\t".join([str(v).ljust(spacer) for v in row]))

    for row in records:
        print("\t".join([repr(x).ljust(spacer) for x in row]))


@click.command()
def setup():
    cwd = pathlib.Path(__file__).parent

    migrate_dir = cwd / "migrate_dir"
    migrate_file = cwd / "migrate.py"
    print(cwd)
    print(migrate_dir)
    print(migrate_file)

cli.add_command(checkdb)
cli.add_command(setup)


def run():
    cli()

if __name__ == "__main__": run()