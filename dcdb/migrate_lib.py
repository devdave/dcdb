
import dcdb
import dataclasses as dcs

from collections import namedtuple

@dcs.dataclass()
class Migration:
    current_stage: str
    ran_at: dcdb.TransformDatetimeType


class Migrator:
    # God module anti-pattern

    def __init__(self, cwd, connection):
        self.cwd = cwd
        self.connection = connection

    def checkdb(self):
        pass

DiffSets = namedtuple("DiffSets", "present,missing")
CompareModelResult = namedtuple("CompareModelResult","all, missing_table, missing_model")

class MISSING:
    #taking a page from dataclasses
    pass

@dcs.dataclass()
class DiffSets:
    in_database: set
    in_module: set

    @property
    def missing_definitions(self):
        return self.in_database - self.in_module

    @property
    def missing_database(self):
        return self.in_module - self.in_database

    @property
    def present(self):
        return self.in_module & self.in_database

    @property
    def no_missing(self):
        return len(self.in_database ^ self.in_module) == 0


class Schema:

    def __init__(self, connection):
        self.connection = connection

    @property
    def tables(self):
        return [row['name'] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]



    def compare(self, scope):

        tables = set(self.tables)
        module_names = set()
        for name in scope.__dict__.keys():
            if name[0] == "_": continue
            if hasattr(getattr(scope, name), "_NO_BIND_") is True: continue
            if hasattr(getattr(scope, name), "__dataclass_fields__") is False: continue
            module_names.add(name)


        return DiffSets(tables, module_names)


    def compare_model(self, model) -> CompareModelResult:
        model_fnames = set(model._meta_.fields.keys())

        table_info = {}
        sql = f"PRAGMA table_info({model._meta_.name})"
        for row in self.connection.execute(sql):
            table_info[row['name']] = {k:row[k] for k in row.keys()}

        table_fnames = set(table_info.keys())

        all_known = table_fnames | model_fnames
        table_missing = model_fnames - table_fnames
        model_missing = table_fnames - model_fnames


        return CompareModelResult(all_known, table_missing, model_missing)


    def add_column(self, model, column_name, column_type, default_val = MISSING ):

        assert column_name in model._meta_.fields.keys()
        assert column_name not in self.compare_model(model).missing_model

        if default_val is MISSING:
            self.connection.execute("ALTER TABLE {model._meta_.schema_name} ADD COLUMN {column_name} {column_type} NOT NULL")
        else:
            if default_val is None:
                default_val = "null"

            self.connection.execute(
                "ALTER TABLE {model._meta_.schema_name} ADD COLUMN {column_name} {column_type} DEFAULT {default_val}")


