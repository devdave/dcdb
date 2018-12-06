# pylint: disable=C0103
"""
    DB DataClass abstraction layer

    Version: Super Alpha


    turns
    @dataclass
    class Something:
        name: str
        age: int

    into a sqlite table `Something` with columns name and age


    Style rules:
        all variables are defined at the start of methods and functions to make it easier for a symbolic debugger
            watch to catch the moment the variable is changed.

        aggregate callables are not required for unit-testing but the calls they make MUST be tested.
"""
from __future__ import annotations

import sys
import dataclasses as dcs
import sqlite3
import contextlib
import pickle
import abc  # TODO is this needed?
import collections
import enum

import inspect

# Avoid application code from having to import sqlite3
IntegrityError = sqlite3.IntegrityError


def cast_from_database(value: object, value_type: type):
    """

    :param value:
    :param field:
    :return:
    """

    debug = value

    if value == "None":
        # TODO figure out how/why this is possible
        retval = None
    elif value is None:
        retval = None

    elif issubclass(value_type, enum.Enum):
        # TODO assuming using int flags
        retval = value_type(int(value))
    elif value_type == str and isinstance(value, str):
        retval = value
    elif value_type in [int, str]:
        retval = value_type(value)
    elif value_type == bool:
        retval = bool(int(value))
    elif hasattr(value_type, "From"):
        retval = value_type.From(value)
    else:
        ex_msg = f"""
            f"Unable to transform {value_type!r} as returned from DB             
            value = {value!r} 
            debug <= {debug!r} 
            field = {field!r}"
        """
        raise ValueError(ex_msg)

    return retval


def cast_to_database(value, value_type: type):
    debug = value

    if value is None:
        retval = None
    elif issubclass(value_type, enum.Enum):
        retval = value.value
    elif value_type == bool:
        retval = int(value)
    elif value_type in [str, int]:
        retval = value_type(value)
    elif hasattr(value_type, "To"):
        retval = value_type.To(value)
    else:
        ex_msg = \
            f"""
                f"Unable to transform {value_type!r} as returned from DB                     
                value = {value!r} 
                debug <= {debug!r} 
                field = {field!r}
            """
        raise ValueError(ex_msg)

    return retval


class AutoCast(metaclass=abc.ABCMeta):
    pass


class AutoCastDict(AutoCast):
    """
        Pickle is used vs json because
        {1:"foo"} becomes {"1":"foo"} is a few cases

        Why pickle versus JSON?
        1 becomes '1' when going from str to json
        https://bugs.python.org/issue32816
    """

    @classmethod
    def From(self, value):
        return pickle.loads(value)

    @classmethod
    def To(cls, value: dict):
        return pickle.dumps(value)


class SQLOperators(enum.Enum):
    AND = "AND"
    OR = "OR"
    NOTEQ = "!="
    LIKE = "LIKE"


class AutoSelect(AutoCast):

    def __init__(self, target_table: str, target_column: str, source_column: str):
        self.__target = None

        self.__target_table = target_table
        self.__target_column = target_column
        self.__source_column = source_column

        super().__init__()

    def __get__(self, owner: DBCommonTable, objtype=None):
        if owner is None:
            return self

        # Let the crazy abuse of semi-private properties begin!
        fk_id = getattr(owner, self.__source_column)
        if fk_id is None:
            # TODO abuse some sort of proxy class to catch owner.autoselect_property.Create to auto assign
            # to this autoselect
            return owner._conn_.t[self.__target_table]

        if self.__target is None:
            self.__target = owner._conn_.t[self.__target_table].Get(f"{self.__target_column}=?", fk_id)

        return self.__target

    def __set__(self, owner, value):

        if isinstance(value, DBCommonTable):
            setattr(owner, self.__source_column, getattr(value, self.__target_column))
        else:
            setattr(owner, self.__source_column, value)

    def __delete__(self, obj):
        setattr(obj, self.__source_column, None)


class ProxyList(list):

    def _set_owner(self, auto_list: AutoList, owner: DBCommonTable) -> None:
        self.auto_list = auto_list
        self.owner = owner

    def __call__(self, *args, **kwargs) -> DBCommonTable:
        print(*args, **kwargs)
        return self.auto_list.create(self.owner, *args, **kwargs)

    def add(self, *records: DBCommonTable) -> DBCommonTable:
        print(records, self.owner, self.auto_list)
        return self.auto_list.add(self.owner, *records)

    def remove(self, record: DBCommonTable) -> DBCommonTable:
        print(record, self.owner, self.auto_list)
        return self.auto_list.remove(self.owner, record)


TableSpec = collections.namedtuple("TableSpec", "name, column")


class AutoList:
    __slots__ = (
        "__parent_table",

        "__child_table",
        "__owner",
        "__conditions",
        "__orderby",
        "__rfetch",
        "__rcreate",
        "__radd",
        "__rremove",
        "__cache")

    def __init__(self, parent_table, child_table, owner=None, conditions=None, orderby=None, creator=None, adder=None,
                 remover=None, __cache=None):

        self.__parent_table = TableSpec(*parent_table) if not isinstance(parent_table, TableSpec) else parent_table
        self.__child_table = TableSpec(*child_table) if not isinstance(child_table, TableSpec) else child_table

        self.__owner = owner
        self.__conditions = conditions
        self.__orderby = orderby
        self.__rcreate = creator
        self.__radd = adder
        self.__rremove = remover
        self.__cache = __cache

    def where(self, *joins):

        computed = []
        str_operators = {"AND", "OR"}  # TODO add more

        def single_term(term):
            return lambda p, c: term.format(parent=p, child=c)

        def multi_term(sql, *arg):
            return lambda parent, child: term[0].format(*term[1:], parent=parent, child=child)

        operator_added = False
        for pos, term in enumerate(joins):

            term_len = len(term)
            new_component = None

            if isinstance(term, str) is True:
                operator_added = term in str_operators
                new_component = single_term(term)

            elif term_len == 2:
                # "lvalue={0}", some_constant_value
                new_component = multi_term(term[0], term[1])
            elif term_len >= 3:
                new_component = multi_term(term[0], *term[1:])

            if pos != 0 and pos % 2 != 0 and operator_added is False:
                # TODO a lot more unit-tests on this because of how goofy it feels
                computed.append(lambda p, c: "AND")
                operator_added = False

            computed.append(new_component)

        if len(computed) == 2:
            # TODO a lot more unit-tests on this because of how goofy it feels
            computed = computed[:1]

        self.__conditions = computed

    def orderby(self, order_str):
        self.__orderby = order_str

    def creator(self, func):
        return type(self)(
            self.__parent_table,
            self.__child_table,
            self.__owner,
            self.__conditions,
            self.__orderby,
            func,
            self.__radd,
            self.__rremove,
            self.__cache
        )

    def create(self, parent_table, *args, **kwargs) -> DBCommonTable:
        self.__cache = None
        child_table = self.__owner.t[self.__child_table.table]
        if self.__rcreate is not None:
            return self.__rcreate(self.__owner, child_table, **kwargs)
        else:
            kwargs[self.__child_table.column].setdefault(self.__owner[self.__parent_table.column])
            return child_table.Insert(**kwargs)

    def adder(self, func):
        return type(self)(
            self.__parent_table,
            self.__child_table,
            self.__owner,
            self.__conditions,
            self.__orderby,
            self.__rcreate,
            func,
            self.__rremove,
            self.__cache
        )

    def add(self, parent_table, *tables: [DBCommonTable]) -> DBCommonTable:
        self.__cache = None
        if self.__radd is not None:
            [self.__radd(self.__owner, parent_table, table) for table in tables]
        else:
            for table in tables:
                table[self.__child_table.column] = parent_table[self.__parent_table.column]
                table.update()

    def remover(self, func):
        return type(self)(
            self.__parent_table,
            self.__child_table,
            self.__owner,
            self.__conditions,
            self.__orderby,
            self.__rcreate,
            self.__radd,
            func,
            self.__cache
        )

    def remove(self, parent_table, *records) -> DBCommonTable:
        self.__cache = None
        if self.__rremove is not None:
            [self.__rremove(self.__owner, parent_table, record) for record in records]
        else:
            for record in records:
                record[self.__child_table.column] = None
                record.update()

    def __get__(self, obj, objtype=None):

        if obj is None:
            return self
        else:
            self.__owner == obj

        if self.__cache is None:
            child_table = obj.tables[self.__child_table.name]
            where = " ".join([term(obj, child_table.bound_cls) for term in self.__conditions])
            records = ProxyList(child_table.Select(where).fetchall())
            records._set_owner(self, obj)
            self.__cache = records
        else:
            records = self.__cache

        return records


@dcs.dataclass
class ColumnDef:
    """
        TODO make ABC interface?
    """
    python: object
    database: str

    def From(self, intermediate):
        return intermediate

    def To(self, intermediate):
        return intermediate


@dcs.dataclass
class TableDef:
    """
        Used during the construction/create table phase to
        add index/constraints after column definitions.
    """
    pass


class DBConnection:
    """
        Database connection manager

        Responsible for keeping a reference/handle of a sqlite3 connection
        and a table registry (which maps to said database)
    """

    _sql_list_tables = """"
    SELECT name, sql
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name;"""

    _sql_describe_table = """
        PRAGMA table_info(%s)
    """

    def __init__(self, dburl: str = None):

        self.dburl = dburl

        if dburl is not None:
            self._conn_ = sqlite3.connect(str(dburl))
            self._conn_.row_factory = sqlite3.Row
            self._conn_.isolation_level = None

        self.tables = self.t = self.registry = TablesRegistry(self)
        self._dirty_records = set()
        self._dirty_records_track_changes = False


    def close(self):
        del self.tables
        del self.t
        del self.registry
        del self._dirty_records
        self._conn_.close()

    def _dirty_record_track_changes(self, flag: bool):
        self._dirty_records_track_changes = flag
        if flag is False:
            self._dirty_records = set()

    @property
    def dirty_records_tracking(self):
        return self._dirty_records_track_changes

    def _dirty_record_add(self, record: DBCommonTable):
        if self._dirty_records_track_changes:
            self._dirty_records.add(record)
            return True
        else:
            return False

    def _dirty_record_remove(self, record: DBCommonTable):
        if record in self._dirty_records:
            self._dirty_records.remove(record)

    @contextlib.contextmanager
    def track_changes(self):
        try:
            self._dirty_record_track_changes(True)
            yield
        finally:
            for table in {x for x in self._dirty_records}:
                table.update()
                table._dirty_reset()
            self._dirty_record_track_changes(False)

    def binds(self, *tables: [DBCommonTable], create_table: bool = True):
        return [self.bind(table, create_table=create_table) for table in tables]

    def bind(self, table, create_table: bool = True):
        bound_class = self.registry.mk_bound_dataclass(table, table.__name__)

        if create_table:
            DBSQLOperations.Create_table(self, bound_class, bound_class._table_)

        return bound_class

    def bind_scan(self, scope, ignore: list = None, create_table = True):
        ignore = [] if ignore is None else ignore

        for name in [name for name in dir(scope) if name.startswith("_") is False and name not in ignore]:
            if inspect.isclass(getattr(scope, name, None)):
                if(hasattr(getattr(scope, name, None), "__dataclass_fields__")):
                    self.bind(getattr(scope, name, None), create_table=create_table)


    def handle(self):
        return self._conn_

    @property
    def direct(self):
        return self._conn_

    @property
    def cursor(self):
        return self._conn_.cursor()

    @property
    def execute(self):
        return self.cursor.execute


class DBRegisteredTable:
    #TODO kill this off and or consolidate somehow
    __slots__ = ('connection', 'bound_cls', 'table_name', 'fields')

    def __init__(self, connection, bound_cls, table_name, fields):
        self.connection = connection
        self.bound_cls = bound_cls
        self.table_name = table_name
        self.fields = fields

    def __getattr__(self, item):
        return getattr(self.bound_cls, item)

    def __call__(self, *args, **kwargs):
        if "id" not in kwargs:
            return self.bound_cls(None, *args, **kwargs)
        else:
            return self.bound_cls(*args, **kwargs)

    def Insert(self, **kwargs):
        return self.bound_cls.Create(**kwargs)

    def Insert_many(self, *column_sets):
        collection = []
        for column_fields in column_sets:
            try:
                collection += [self.bound_cls.Create(**column_fields)]
            except sqlite3.IntegrityError as ex:
                raise ValueError(f"IntegrityError on {column_fields} - {ex}")
            except Exception as ex:
                raise ValueError(f"{type(ex)} excception on {column_fields} - {ex}")

        return collection



    def Count(self, where=None, *args, **kwargs):
        kwargs['params'] = "COUNT(*)"
        cursor = DBSQLOperations.Select(self.connection, self.table_name, where, *args, columns="COUNT(*)")
        return cursor.fetchone()[0]



    def Update(self, where, **values):
        return DBSQLOperations.Update(self.connection, self.table_name, self.fields, where, **values)


class DBSQLOperations:

    @classmethod
    def Create_table(cls, connection, table_cls, table_name, safeguard_if_not_exists: bool = True):
        """
            table_definitions - None column definitions like foreign keys and constraints

        """
        body = ""
        column_map = dict()
        type_map = {int: "INTEGER", str: "TEXT", bool: "BOOLEAN"}

        table_def_match = lambda field_val: field_val.type is TableDef and isinstance(field_val.default, str)
        table_definitions = {f.name: f.default for f in dcs.fields(table_cls) if table_def_match(f)}

        for field in dcs.fields(table_cls):
            if field.name == "id": continue  # this is ALWAYS part of the def so don't bother
            if field.type is None: continue
            if field.type is TableDef: continue
            column_map[field.name] = cls._Create_column(field, type_map)

        sql = [
            f"""CREATE TABLE {"IF NOT EXISTS" if safeguard_if_not_exists else ""} {table_name} (""",
            "id INTEGER PRIMARY KEY NOT NULL, "
        ]

        sql.append(", ".join([f"{k} {v}" for k, v in column_map.items()]), )

        if table_definitions:
            sql.append(", ")
            sql.append(", ".join([str(td) for td in table_definitions.values()]))

        sql = " ".join(sql).strip() + ")"

        try:
            return connection.execute(sql)
        except sqlite3.OperationalError as ex:
            print(ex)
            print("Failed query:\n", f"{sql!r}:{len(sql)}")
            raise

    @classmethod
    def _Create_column(cls, column_field: dcs.Field, type_map: {type: str}) -> str:
        # TODO make a test just for this method

        default_value = dcs.MISSING
        sql_column = None

        def find_default(column_field):
            return column_field.default_factory() \
                    if column_field.default_factory is not dcs.MISSING \
                    else column_field.default



        if isinstance(column_field.default, ColumnDef):
            return column_field.default.database

        # TODO find a better way to handle this
        if column_field.type not in type_map:
            if issubclass(column_field.type, AutoCastDict):
                sql_column = "TEXT"
                default_value = find_default(column_field)

            elif issubclass(column_field.type, enum.Enum):
                if issubclass(column_field.type, enum.IntEnum):
                    sql_column = "INT"
                else:
                    sql_column = "TEXT"

                default_value = find_default(column_field)

            else:
                sql_column = "TEXT"
        else:
            sql_column = type_map[column_field.type] if column_field.type in type_map else "TEXT"

            default_value = find_default(column_field)

        if default_value is not dcs.MISSING:
            if default_value is None:
                default_value = "Null"
            else:
                default_value = cast_to_database(default_value, column_field.type)

            sql_column += f" DEFAULT {default_value}"
        else:
            sql_column += " NOT NULL"

        # TODO

        return sql_column

    @classmethod
    def Insert(cls, connection, table_name, dc_fields, **column_fields):

        if column_fields:
            keys = list(column_fields.keys())
            sql = [f"INSERT INTO {table_name}",
                   "(", ", ".join(keys), ")"
                                         "VALUES",
                   "(", ", ".join(["?" for x in range(0, len(keys))]), ")"
                   ]
            sql = " ".join(sql).strip()

            values = [cast_to_database(cf_value, dc_fields[cf_name].type) for cf_name, cf_value in
                      column_fields.items()]

            return connection.execute(sql, values)

        else:
            sql = f""" INSERT INTO {table_name} DEFAULT VALUES"""
            return connection.execute(sql)

    @classmethod
    def Select(cls, connection, table_name, where, *where_vals, **params):
        sql = f"""
            SELECT
                {params['columns'] if "columns" in params else '*'}
            FROM {table_name}
            {"WHERE" if where else ""}
                {where if where else ''}
            {params['append'] if 'append' in params else ""}
        """.strip()

        if where_vals:
            assert where_vals is not None, f"Where has ? but no arguments for {where}!"
            return connection.execute(sql, where_vals)
        else:
            return connection.execute(sql)

    @classmethod
    def Update(cls, connection, table_name, dc_fields, where, **values: dict):

        field_values = {}
        params = {k[1:]: v for k, v in values.items() if k[0] == "_"}

        for k, v in values.items():
            if k.startswith("_"): continue
            field_values[k] = cast_to_database(v, dc_fields[k].type, )

        if isinstance(where, list):
            where_sql = where[0]
            where_values = where[1:]
        else:
            where_values = []
            where_sql = where

        sql = f"""
            UPDATE
                {table_name}
            SET
                {", ".join([f"{k}=?" for k in field_values.keys()])}
            WHERE
                {where_sql}
            {f"LIMIT {params['limit']}" if "limit" in params else ""}
            {params['append'] if 'append' in params else ""}
        """

        # print(sql, list(field_values.values()) + list(where_values))
        return connection.execute(sql, list(field_values.values()) + list(where_values))

    @classmethod
    def Delete(cls, connection, table_name, where: str, where_vals: list = None, limit: int = 1):

        assert where, f" Where {where!r} is empty, use '1=1' for a total table purge"
        sql = f"""
            DELETE FROM
                {table_name}
            WHERE {where}
        """
        # TODO apparently my sqlite build does NOT have order by/limit support with DELETE
        # this slightly disturbs me
        # https://www.sqlite.org/compile.html#enable_update_delete_limit

        # print(repr(sql), repr(where), repr(where_vals), repr(limit))
        if where_vals:
            return connection.execute(sql, **where_vals)
        else:
            return connection.execute(sql)


class DBDirtyRecordMixin:

    def _init_dirty_record_tracking_(self):
        self._dirty_record = False
        self._dirty_record_ignore = True

    def __hash__(self):
        """
        Hashes records by their garbage collection reference ID.

        Only obvious edge case is if I have two separate instances of the same record and
        they both get caught up in the dirty records list.   If I have record A and B of the same database record
        but B is updated after A, changes to A will be lost.   Otherwise I am hard pressed to see any other immediate
        problems.


        :return: hash()
        """

        return hash(id(self))

    @property
    def _is_dirty(self):
        return self._dirty_record

    def _set_dirty(self):
        if self._conn_._dirty_record_add(self):
            object.__setattr__(self, "_dirty_record", True)

    def _dirty_reset(self):
        self._conn_._dirty_record_remove(self)
        object.__setattr__(self, "_dirty_record", False)

    def __setattr__(self, key, value):
        self._set_dirty()
        object.__setattr__(self, key, value)


@dcs.dataclass
class DBCommonTable(DBDirtyRecordMixin):
    """
    :cvar _conn_: DBConnection
    :cvar _table__: str
    :cvar _original_: object
    :cvar _fields_:[dcs.Field]
    """

    id: int  # TODO should I make this an initvar?

    def __post_init__(self):

        self.tables = self._conn_.tables

        # Post processing to convert from DB to Application
        for field in self._fields_.values():
            if field.name == "id": continue

            # avoid mutation tracking as much as possible
            orig_value = getattr(self, field.name)
            value = cast_from_database(orig_value, field.type)
            super().__setattr__(field.name, value)

        # super().__post_init__()
        self._init_dirty_record_tracking_()

    def __setitem__(self, key, value):
        if key in self._fields_:
            setattr(self, key, value)

    def __getitem__(self, key):
        if key in self._fields_:
            return getattr(self, key)
        else:
            raise ValueError(f"No {key} in {self}")

    @classmethod
    def DB(cls):
        return cls._conn_.cursor

    @classmethod
    def Select(cls, where, *args, **kwargs):
        cursor = DBSQLOperations.Select(cls.DB(), cls._table_, where, *args, **kwargs)
        return DBCursorProxy(cls, cursor)

    @classmethod
    def Get(cls, where, *where_vals, **kwargs):
        return cls.Select(where, *where_vals, **kwargs).fetchone()

    @classmethod
    def Create(cls, **kwargs):
        cursor = DBSQLOperations.Insert(cls.DB(), cls._table_, cls._fields_, **kwargs)
        cursor = DBSQLOperations.Select(cls.DB(), cls._table_, f"id={cursor.lastrowid}")
        return DBCursorProxy(cls, cursor).fetchone()

    def delete(self):
        cursor = DBSQLOperations.Delete(self._conn_, self._table_, f"id={self.id}")

        for field in dcs.fields(self):
            setattr(self, field.name, None)
        return cursor

    def save(self):
        values = dcs.asdict(self)
        rec_id = values.pop("id", None)

        retval = None

        if rec_id is not None:
            retval = DBSQLOperations.Update(self._conn_, self._table_, self._fields_, f"id={rec_id}", **values)
        else:
            retval = DBSQLOperations.Insert(self._conn_, self._table_, self._fields_, **values)
            self.id = retval.lastrowid

        self._dirty_reset()

    def update(self):
        values = dcs.asdict(self)
        rec_id = values.pop("id")
        cursor = DBSQLOperations.Update(self._conn_, self._table_, self._fields_, f"id={rec_id}", **values)
        self._dirty_reset()
        return cursor.rowcount == 1


class DBCursorProxy:
    """
        Wrap around the cursor instance so that record
        requests are instantiated into the appropriate

    """

    def __init__(self, factory: DBCommonTable, cursor: sqlite3.Cursor):
        self._factory = factory
        self._cursor = cursor

    def fetchone(self) -> DBCommonTable:
        row = self._cursor.fetchone()
        if row is None: return None

        return self._factory(**row) if row is not None else None

    def fetchmany(self, size=sqlite3.Cursor.arraysize) -> DBCommonTable:
        for row in self._cursor.fetchmany(size):
            yield self._factory(**row) if row is not None else None

    def fetchall(self) -> DBCommonTable:
        for row in self._cursor.fetchall():
            yield self._factory(**row) if row is not None else None

    def __getattr__(self, key):
        return getattr(self._cursor, key)


class TablesRegistry:
    """

        Responsible for converting dataclasses into DBDataclasses and keeping
        a registry of these new classes.

        TODO, creates a new dataclass and then creates a slotted version.   Perhaps
        a reconnect mechanism would be a good idea to save the class definitions for
        environments like flask where bind maybe called more than once.


    """

    def __init__(self, connection):
        self._registry: {str: DBCommonTable} = dict()
        self._connection = connection

    def mk_bound_dataclass(self, source_cls, name: str) -> DBCommonTable:
        """

        :param source_cls: a dataclass intended to be used for working with a databasee
        :param name:  Generally is cls.__name__
        :return: <type cls name=name, MRO=(cls,DBCommonTable>
        """

        def eval_type(st):
            return st if not isinstance(st, str) else eval(st, vars(sys.modules[self.__class__.__module__]))

        new_dataclass_fields: [str, type, dcs.Field] = []
        dataclass_fields: [dcs.Field] = dcs.fields(source_cls)
        # TODO why did I do this twice?
        for df in dataclass_fields:
            df.type = eval_type(df.type)

        # TODO put this in a higher scope or use a common heritage sentinel class
        exclusion_list = [TableDef]

        for source_field in [f for f in dataclass_fields if f.type not in exclusion_list]:
            new_field = dcs.field(default=source_field.default, default_factory=source_field.default_factory)

            new_dataclass_fields.append((
                source_field.name,
                eval_type(source_field.type),
                new_field))

        db_cls = dcs.make_dataclass(f"DCDB_{name}", new_dataclass_fields, bases=(source_cls, DBCommonTable))
        db_cls_fields = dcs.fields(db_cls)
        for field in db_cls_fields:
            field.type = eval_type(field.type)

        # TODO - Figure out why __hash__ from DCCommonTable is being stripped out
        setattr(db_cls, "__hash__", lambda instance: hash(id(instance)))
        db_cls = self.ioc_assignments(db_cls, source_cls, name, db_cls_fields)

        self._registry[name] = DBRegisteredTable(self._connection, db_cls, name, db_cls._fields_)
        return db_cls

    def generate_slotted_dcdb(self, db_cls: DBCommonTable):

        """"

            Debatable if this is worth it or not.
        """

        # sourced from https://github.com/ericvsmith/dataclasses/blob/master/dataclass_tools.py

        cls_dict = dict(db_cls.__dict__)
        field_names = [f.name for f in dcs.fields(db_cls)]
        field_names += ['_conn_', '_table_', '_original_', '_fields_']
        field_names += ["id", "_dirty_record", "_dirty_record_ignore"]

        cls_dict['__slots__'] = field_names
        for field_name in field_names:
            cls_dict.pop(field_name, None)

        cls_dict.pop('_dict__', None)
        qualname = getattr(db_cls, '__qualname__', None)
        new_cls = type(db_cls)(db_cls.__name__, db_cls.__bases__, cls_dict)
        if qualname is not None:
            new_cls.__qualname__ = qualname

        return new_cls

    def ioc_assignments(self, db_cls: DBCommonTable, source_cls, name, db_cls_fields) -> DBCommonTable:

        setattr(db_cls, "_conn_", self._connection)
        setattr(db_cls, "_table_", name)
        setattr(db_cls, "_original_", source_cls)
        setattr(db_cls, "_fields_", {f.name: f for f in db_cls_fields})

        return db_cls

    def get_table(self, table_name: str) -> DBCommonTable:
        return self._registry.get(key)

    def __getattr__(self, key):
        return self._registry.get(key)

    def __getitem__(self, key):
        return self._registry.get(key)