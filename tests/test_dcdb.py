# pylint: disable=C0111,C0112,C0301,C0103,W0621,C0102
# TODO remake .pylintrc?  I only want some of these disables for tests though
"""
    Tests for DCDB
"""

import logging
import pytest

import dcdb

from dataclasses import dataclass, fields, field
import enum
import pathlib

import datetime as dt

import typing

LOG = logging.getLogger(__name__)


@pytest.fixture
def connection(request) -> dcdb.DBConnection:
    """
        Fixture used to speed up testing
    """
    db_file = pathlib.Path(__file__).parent / "db" / f"{request.function.__name__}.sqlite"
    if db_file.exists():
        db_file.unlink()

    return dcdb.DBConnection(str(db_file))


@pytest.fixture()
def conn2(connection: dcdb.DBConnection):
    """

    :param connection: dcdb.DBConnection uses the connection fixture
    :return:
    """

    @dataclass()
    class Widget:
        name: str
        age: int
        panacea: bool

    connection.bind(Widget)

    return connection


@dataclass
class Widget:
    # __table__ = "widgets"
    name: str
    age: int
    panacea: bool = False  # in retrospect this makes no sense but I am just going to keep using this word


def test_dcdb_cast_to_database_type__fix_ordering_error():
    """
    Odd test but written to isolate a bug

    """

    value = "Bob"
    result = dcdb.cast_to_database(value, str)
    assert isinstance(result, str)
    assert value == result


def test_dcdb__cast_to_database_AND_cast_from_database__handle_datetime(connection):
    import datetime

    @dataclass()
    class TimePiece:
        a_date: datetime.date
        a_time: datetime.time
        a_datetime: datetime.datetime

    connection.bind(TimePiece)

    test_datetime = datetime.datetime(1955, 11, 5, 11, 10, 54, 67345)
    test_date = datetime.date(1941, 12, 7)
    test_time = datetime.time(10, 20, 11)

    record = connection.t.TimePiece(a_date=test_date, a_datetime=test_datetime, a_time=test_time)

    assert record.a_date == test_date
    assert record.a_time == test_time
    assert record.a_datetime == test_datetime


def test_DBConnection___connects():
    my_conn = dcdb.DBConnection(":memory:")
    result = my_conn.handle().execute("SELECT 1=1").fetchone()
    assert result[0] == 1


def test_FieldPickled__AND__dcdb_cast_to_database___works():
    import pickle

    @dataclass
    class Foo:
        colors: dcdb.FieldPickled

    test_value = {"red": 1, "blue": 2, "complicated": [1, 2, "this isn't a color"]}
    test_subject = Foo(test_value)

    foo_fields = fields(Foo)
    colors_field = fields(Foo)[0]
    result = dcdb.cast_to_database(test_value, colors_field.type)
    assert isinstance(result, bytes)
    assert result == pickle.dumps(test_value)


def test_dcdb__cast_to_database():
    class Switch(enum.Enum):
        OFF = 0
        ON = 1

        @classmethod
        def To(cls, value, _):
            return value.name

        @classmethod
        def From(cls, value, _):
            return cls.__members__[value]

    test = Switch.OFF
    expected = "OFF"
    actual = dcdb.cast_to_database(test, Switch)
    assert expected == actual


def test_DBTableRegistry___mk_bound(connection):
    @dataclass()
    class Widget:
        name: str
        age: int
        panacea: bool

    connection.bind(Widget)
    res = connection.handle().execute(
        "SELECT name FROM sqlite_master WHERE type='table' and name = 'Widget' ORDER BY name;")
    name = res.fetchone()
    assert name[0] == "Widget"
    assert connection.registry.Widget is not None
    rows = connection.execute("PRAGMA table_info(Widget)").fetchall()

    assert rows[0]['name'] == "id"
    assert rows[0]['type'] == "INTEGER"

    assert rows[1]['name'] == "name"
    assert rows[1]['type'] == "TEXT"

    assert rows[2]['name'] == "age"
    assert rows[2]['type'] == "INTEGER"

    # Sqlite coerces bools to intege
    # which is kind of a stretch as basically sqlite coerces everything into a string

    assert rows[3]['name'] == "panacea"
    assert rows[2]['type'] == "INTEGER"


def test_DBCommonTable___ishashable(conn2):
    x = conn2.t.Widget(name="Bob", age=10, panacea=True)
    y = conn2.t.Widget(name="Alive", age=12, panacea=False)
    x2 = conn2.t.Widget.Get("name=?", "Bob")
    assert (y == x) is False
    assert (x == x) is True
    assert (x == x2) is True
    records = set()
    assert x.__hash__ is not None
    records.add(x)
    records.add(y)


def test_DBConnection__bind_blowsup_with_typeerror_on_nondataclass(connection):
    class Bogus:
        foo: str = "bar"

    with pytest.raises(TypeError):
        connection.bind(Bogus)


def test_DBConnection___binds_multiple_tables(connection):
    @dataclass()
    class Foo:
        name: str

    @dataclass()
    class Bar:
        name: str
        num: int

    @dataclass()
    class Bob:
        last_name: str
        age: int

    (MyFoo, MyBar, MyBob,) = connection.binds(Foo, Bar, Bob)

    for (test_cls, db_cls,) in ((Foo, MyFoo,), (Bar, MyBar,), (Bob, MyBob,)):
        assert issubclass(db_cls, test_cls)
        assert issubclass(db_cls, dcdb.DBCommonTable)


def test_DBCommonTable__assure_post_init_overload_works(connection):
    @dataclass()
    class Foo:
        def __post_init__(self, *args, **kwargs):
            super().__post_init__(*args, **kwargs)
            LOG.debug("I was called")

    connection.bind(Foo)

    record = connection.tables.Foo()


def test_make_complex_widget(connection):
    from typing import ClassVar
    @dataclass
    class ComplexWidget:
        # by definition of how dataclasses work, these cannot be empty/unassigned/null so don't bother telling sqlite to enforce that.
        #
        name: str
        age: int

        NameAgeUnique: dcdb.TableDef = "CONSTRAINT NameAgeUnique UNIQUE(name,age)"

    connection.bind(ComplexWidget)
    res = connection.handle().execute("""PRAGMA index_info(sqlite_autoindex_ComplexWidget_1)""")

    index_rows = res.fetchall()
    """
        [0, 1, 'name']
        [1, 2, 'age']
    """
    assert len(index_rows) == 2
    assert index_rows[0][2] == "name"
    assert index_rows[1][2] == "age"


def test_db_dataclass_throws_error_on_missing_param(conn2):
    with pytest.raises(ValueError):
        conn2.t.Widget(id=1)

    with pytest.raises(ValueError):
        conn2.t.Widget(id=1, name="Bob")

    with pytest.raises(ValueError):
        conn2.t.Widget(id=1, name="Bob", age=55)

    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget()

    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget(name="Bob")

    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget(name="Bob", age=55)

    record = conn2.t.Widget(name="Bob", age=55, panacea=True)
    assert record.id == 1  # asserts this is the first record added to the table, therefore assumes only record
    assert record.name == "Bob"
    assert record.age == 55
    assert record.panacea == True


def test_DBTableRegistry__create(conn2):
    with pytest.raises(dcdb.IntegrityError):
        conn2.t.Widget(name="Bob", age=55)

    conn2.t.Widget(name="Bob", age=55, panacea=False)
    row = conn2.execute("SELECT * FROM Widget LIMIT 1").fetchone()

    assert row['id'] == 1
    assert row['name'] == "Bob"
    assert row['age'] == 55


def test_DBTableProxy_InsertMany(conn2):
    MyWidget = conn2.t.Widget
    MyWidget.Insert_many(
        dict(name="Bob", age=10, panacea=True),
        dict(name="Joe", age=22, panacea=False),
        dict(name="Steve", age=35, panacea=True)
    )

    record = conn2.execute("SELECT COUNT(*) FROM Widget").fetchone()
    assert record[0] == 3

    bob = MyWidget.Get("name=?", "Bob")
    Joe = MyWidget.Get("name=?", "Joe")
    Steve = MyWidget.Get("name=?", "Steve")
    assert bob.age == 10
    assert Joe.name == "Joe"  # this seems redundent


def test_DBCommonTable_AND_DBRegistry__create_with_default_factory(connection):
    @dataclass()
    class Test:
        foo: str = "Hello"
        bar: int = dcdb.ColumnDef(python=None, database="TIMESTAMP DEFAULT (strftime('%s','now'))")

    connection.bind(Test)

    instance = connection.t.Test.Create()


def test_DBTableRegistry__create(conn2):
    with pytest.raises(ValueError):
        record = conn2.t.Widget(id=1)

    with pytest.raises(dcdb.IntegrityError):
        record = conn2.t.Widget(name="Bob", age=55)

    record = conn2.t.Widget(name="Bob", age=55, panacea=True)
    record.save()
    assert record.id == 1
    assert record.name == "Bob"
    assert record.age == 55
    assert record.panacea == True

    record = conn2.t.Widget(name="Bob", age=55, panacea=False)
    record.save()
    assert record.id == 2
    assert record.panacea == False


def test_DBConnection__data_integrity_from_sqlite(conn2):
    with pytest.raises(dcdb.IntegrityError):
        conn2.execute("INSERT INTO Widget(name,age) VALUES ('Bob', 33)")


def test_DBCommonTable__Select(conn2):
    conn2.execute("INSERT INTO Widget(name,age,panacea) VALUES ('Bob', 33, 1)")
    row = conn2.t.Widget.Select("name=?", "Bob").fetchone()

    assert row.name == "Bob"
    assert row.age == 33
    assert row.panacea == 1


def test_DBRegisteredTable_AND_DBCommonTable___Insert_Select_Get(conn2):
    MyWidget = conn2.t.Widget
    MyWidgetClass = MyWidget.bound_cls

    MyWidget(name="Bob", age=44, panacea=False)

    record = MyWidget.Select("name=?", "Bob").fetchone()

    assert record.name == "Bob"
    assert record.age == 44
    assert record.panacea == False

    record2 = MyWidget.Get("name=?", "Bob")
    assert isinstance(record2, MyWidgetClass)
    assert record == record2


def test_DBCommonTable_update(conn2):
    conn2.bind(Widget)

    # TODO should I consider it a problem that setting id skips saving the actual record?
    #  record = conn2.t.Widget(id=10, name="Joe", age=22, panacea=True)
    record = conn2.t.Widget(name="Joe", age=22, panacea=True)
    assert record.name == "Joe"
    assert record.age == 22
    assert record.panacea is True

    record.panacea = False
    record.age = 33
    record.update()
    same_record = conn2.t.Widget.Get("id=?", record.id)
    assert isinstance(same_record, conn2.t.Widget.bound_cls)
    assert same_record.panacea is False
    assert same_record.age == 33
    assert same_record.id == record.id


def test_DBCommonTable_Update__bulk_update(conn2):
    # Populate the db
    conn2.t.Widget(name="Bob", age=13, panacea=True)
    conn2.t.Widget(name="Joe", age=22, panacea=True)
    conn2.t.Widget(name="Sue", age=34, panacea=True)
    conn2.t.Widget(name="Mary", age=44, panacea=True)
    conn2.t.Widget(name="It", age=9999, panacea=True)

    # Bulk update roughly half
    cursor = conn2.t.Widget.Update(["age<?", 35], panacea=False)

    # Sanity check they are correct
    assert conn2.t.Widget.Get("name=?", "Bob").panacea is False
    assert conn2.t.Widget.Get("name=?", "Joe").panacea is False
    assert conn2.t.Widget.Get("name=?", "Sue").panacea is False
    assert conn2.t.Widget.Get("name=?", "Mary").panacea is True
    assert conn2.t.Widget.Get("name=?", "It").panacea is True


def test_TransformType__works(connection):
    pass


def test_FieldPickled_AND_FieldJSON__works_as_expected(connection):
    @dataclass
    class Foo:
        bar: dcdb.dcdb.FieldPickled
        name: str

    @dataclass()
    class Bar:
        name: str
        blank: dcdb.FieldJSON = None

    connection.bind(Foo)
    connection.bind(Bar)

    test_value = {"a": 1, 2: "b", "three": "c"}
    expected_json = {"a": 1, '2': "b", "three": "c"}

    record = connection.t.Foo(bar=test_value, name="Test thing")

    assert record.bar == test_value, repr(record.bar)

    record2 = connection.t.Bar(name="Bob")

    assert record2.blank == None

    record3 = connection.t.Bar(name="Bob", blank=test_value)

    assert record3.blank == expected_json


def test_DBCommonTable_hashing_equals(conn2):
    record1 = conn2.t.Widget(name="Alice", age=35, panacea=False)
    record2 = conn2.t.Widget.Get("name=?", "Alice")
    record3 = conn2.t.Widget.Get("age=?", 35)
    assert record1.id == record2.id
    assert record1 == record2
    assert record2 == record3
    assert record3 == record1

    @dataclass()
    class Foo:
        height: int
        width: int

    conn2.bind(Foo)
    record4 = conn2.t.Foo(height=12, width=22)
    record5 = conn2.t.Foo.Get("height=?", 12)
    record6 = conn2.t.Foo.Get("width=?", 22)
    assert record4 != record1
    assert record5 != record2
    assert record6 != record3
    assert record6 == record4
    assert record4 == record5
    assert record5 != record3


def test_DBTableRegistry___works(connection):
    MyWidget = connection.bind(Widget)
    SameWidget = connection.t.Widget
    assert issubclass(MyWidget, Widget)
    assert issubclass(SameWidget.bound_cls, Widget)


def test_DBTablesRegistry___exception_on_missing_table(connection):
    with pytest.raises(RuntimeError):
        connection.t.NotATable


def test_DBTableRegistry___tables_direct_children(connection):
    @dataclass()
    class Parent:
        name: str

        @property
        def children(self):
            return self.tables.Child.Select(f"parent_id={self.id}")

        def create_new_child(self):
            child = self.tables.Child(parent_id=self.id)
            child.update()
            return child

        def disown(self, child):
            child.delete()

    @dataclass()
    class Child:
        parent_id: int

        def __str__(self):
            return "Child of {self.parent_id} #{self.id}"

        @property
        def parent(self):
            return self.tables.Parent.Get("id=?", self.parent_id)

    MyParent = connection.bind(Parent)
    MyChild = connection.bind(Child)

    bob = connection.t.Parent(name="Bob")
    bob.create_new_child()
    bob.create_new_child()
    bob.create_new_child()

    children = connection.execute("SELECT * FROM Child").fetchall()
    assert len(children) == 3

    redheaded_stepchild = connection.t.Child.Get("id=2")
    assert redheaded_stepchild.parent.name == "Bob"
    bob.disown(redheaded_stepchild)

    children = connection.execute("SELECT * FROM Child").fetchall()
    assert len(children) == 2


def test_RelationshipFields_DOT_local_one_to_one___works(connection):
    @dataclass()
    class TargetTable:
        name: str
        some_num: int


    @dataclass()
    class OwnerTable:
        name: str
        other_num: int
        target_id: int = None

        target = dcdb.RelationshipFields.local_one_to_one("target_id", "TargetTable.id")


    connection.binds(TargetTable, OwnerTable)

    target_record = connection.t.TargetTable(name="Joe", some_num=1234)
    owner_record = connection.t.OwnerTable(name="Brad", other_num=56)
    useless_owner_record = connection.t.OwnerTable(name="Bungle", other_num=123)


    owner_record.target.set(target_record)

    assert hasattr(owner_record, "target") is True
    assert owner_record.target_id == target_record.id
    assert owner_record.target.name == target_record.name

    owner_record.target.clear()
    assert owner_record.target_id is None


def test_RelationshipFields_DOT_unordered_list__works(connection: dcdb.DBConnection):
    # whoops, realized this isn't tested inside dcdb but in harvester

    @dataclass()
    class Box:


        contents = dcdb.RelationshipFields.unordered_list("Widget", "box_id")

    @dataclass()
    class Widget:
        name: str
        quantity: int
        box_id: int = None

    connection.bind(Box, Widget)

    box = connection.t.Box()
    box.contents.create(name="Nails", quantity=300)
    box.contents.create(name="Hammer", quantity=1)
    box.contents.create(name="Measuring tape", quantity=1)
    box.contents.create(name="Bandaids", quantity=0)

    # NOTE relying heavily on order of inserts
    assert box.contents[1].name == "Hammer"
    assert len(box.contents) == 4
    del box.contents[1]
    assert len(box.contents) == 3
    assert connection.t.Widget.Count() == 3
    assert box.contents[1].name == "Measuring tape"

    with pytest.raises(TypeError):
        box.contents[5] = connection.t.Widget(name="Stuff", quantity=10 ** 5)

    assert len(box.contents) == 3
    assert connection.t.Widget.Count() == 4

    widget = box.contents.first()
    assert widget.name == "Nails"

    widget.delete()

    assert len(box.contents) == 2
    assert box.contents.first().name == "Measuring tape"
    assert box.contents[1].name == "Bandaids"

    bandaids = box.contents.pop(1)
    assert bandaids.quantity == 0
    assert len(box.contents) == 1
    assert connection.t.Widget.Count() == 3


def test_RelationshipFields_DOT_ListSelect___dotted_arguments():
    expected = dcdb.ListSelect("Child", "parent_id")
    dotted = dcdb.ListSelect("Child.parent_id")

    assert expected.child_name == dotted.child_name
    assert expected.relationship_field == dotted.relationship_field

    with pytest.raises(ValueError):
        broken = dcdb.ListSelect("Foo bar")


def test_ReltionshipFields_DOT_dict__by_order(connection: dcdb.DBConnection):
    # TODO should refactor to have a stack based dictionary class because
    # otherwise this behavior is a tad confusing.

    @dataclass()
    class Box:
        widget = dcdb.RelationshipFields.dict("Thing.name", "parent_id", by_order="version")

    @dataclass()
    class Thing:
        name: str
        version: int
        parent_id: int = None

    connection.binds(Box, Thing)

    b = connection.t.Box()

    object1 = b.widget.create(name="Foo", version=1)
    object2 = b.widget.create(name="Foo", version=3)
    object3 = b.widget.create(name="Foo", version=2)

    assert b.widget['Foo'].id == 2

    object3 = b.widget.create(name="Foo", version=4)

    assert b.widget['Foo'].id == 4


def test_RelationshipFields_dot_dict__works(connection: dcdb.DBConnection):
    @dataclass()
    class House:
        price: float
        name: str
        furniture = dcdb.RelationshipFields.dict("Furniture", "type", "house_id")

    @dataclass()
    class Furniture:
        type: str
        material: str
        quantity: int

        house_id: int = None  # TODO foreign key constraint

    # setup
    connection.binds(House, Furniture)
    house = connection.t.House(price=123.45, name="The Manor")

    # Ensure relationship is bound on direct creation
    house.furniture.create(type="Chair", material="Wood", quantity=10)
    assert len(house.furniture) == 1
    assert house.furniture["chair"] is None
    assert house.furniture["Chair"].material == "Wood"

    # ensure binding is independant
    sofa = connection.t.Furniture(type="Sofa", material="cloth", quantity=2)
    assert len(house.furniture) == 1

    # test count and retrieval
    house.furniture.add(sofa)
    assert len(house.furniture) == 2
    assert house.furniture["Sofa"].quantity == 2

    # verify integrity
    keys = house.furniture.keys()
    assert "Sofa" in keys
    assert "Chair" in keys

    #
    del house.furniture['Chair']
    furniture_count = connection.direct("SELECT count(*) FROM Furniture").fetchone()[0]
    assert furniture_count == 2
    assert len(house.furniture) == 1


def test_RelationshipFields_dict__dotted_argument(connection):
    @dataclass()
    class House:
        price: float
        name: str
        furniture = dcdb.RelationshipFields.dict("Furniture.type", "house_id")

    @dataclass()
    class Furniture:
        type: str
        material: str
        quantity: int

        house_id: int = None  # TODO foreign key constraint

    # setup
    connection.binds(House, Furniture)
    house = connection.t.House(price=123.45, name="The Manor")

    house.furniture.create(type="Bed", material="down", quantity=1)

    assert len(house.furniture) == 1
    assert house.furniture['Bed'].material == "down"


def test_RelationshipFields_Named_Left_Join__works(connection):
    @dataclass()
    class Box:
        # One way from Box.things[str]

        things = dcdb.RelationshipFields.named_left_join(
            "Box2Thing", "box_id", "thing_id",
            "Thing", child_name_field="name")

    @dataclass()
    class Box2Thing:
        box_id: int
        thing_id: int
        idx_Box2Thing: dcdb.TableDef = "CONSTRAINT idx_Box2Thing UNIQUE(box_id, thing_id)"

    @dataclass()
    class Thing:
        name: str
        quantity: int
        material: str
        idx_NameMaterial: dcdb.TableDef = "CONSTRAINT NameMaterial UNIQUE(name, material)"

    connection.binds(Box, Box2Thing, Thing)

    toolbox = connection.t.Box()
    storagebox = connection.t.Box()

    hammers = connection.t.Thing(name="Hammer", quantity=2, material="steel")
    nails = connection.t.Thing(name="Nails", quantity=150, material="iron")
    screwdriver = connection.t.Thing(name="Screwdriver", quantity=2, material="aluminum")
    bits = connection.t.Thing(name="Bits", quantity=12, material="carbon steel")
    tape = connection.t.Thing(name="Tape", quantity=1, material="Plastic")

    toolbox.things += [hammers, nails, screwdriver]
    storagebox.things += [screwdriver, bits, tape]

    assert hammers in toolbox.things
    assert hammers not in storagebox.things

    assert nails not in storagebox.things
    assert nails in toolbox.things

    assert bits not in toolbox.things
    assert bits in storagebox

    assert tape not in toolbox
    assert tape in storagebox

    assert connection.t.Thing.Count() == 5
    assert len(toolbox.things) == 3
    assert len(storagebox.things) == 3

    assert toolbox.things['Nails'].quantity == 150
    assert storagebox.things['Bits'].quantity == 12

    copy_screwdriver = toolbox.things['Screwdriver']
    copy_screwdriver.quantity = 1
    copy_screwdriver.save()
    assert storagebox.things['Screwdriver'].quantity == 1


def test_DBCommonTable___fix_error_columns_mismatch(conn2):
    """
        At one point, the table/record creation mechanism
        used the internal fields dictionary for record creation
        which had the unintended consequence of creating records
        with fields that were not in order and therefore value mismatched.

    """

    @dataclass
    class OtherWidget:
        length: int
        height: int
        weight: int
        name: str

    conn2.bind(OtherWidget)
    thingamajig = conn2.t.OtherWidget(name="Thinga 2000", length=10, height=1, weight=8)
    assert thingamajig.name == "Thinga 2000"
    assert thingamajig.length == 10
    assert thingamajig.height == 1
    assert thingamajig.weight == 8


def test_mutation_tracking(conn2: dcdb.DBConnection):
    @dataclass
    class OtherWidget:
        length: int
        height: int
        weight: int
        name: str

    conn2.bind(OtherWidget)
    bob = conn2.t.Widget(name="Bob", age=44, panacea=False)
    thingamajig = conn2.t.OtherWidget(name="Thinga 2000", length=10, height=1, weight=8)

    assert bob._is_dirty is False
    assert thingamajig._is_dirty is False

    # Known and intended gotcha
    thingamajig.height = 5
    assert thingamajig._is_dirty is False

    with conn2.track_changes():
        bob.age = 22
        assert bob._is_dirty is True
        assert len(conn2._dirty_records) == 1

        thingamajig.weight = 12
        thingamajig.name = "New Hotness 2.0"
        assert len(conn2._dirty_records) == 2
        assert thingamajig._is_dirty is True

    assert len(conn2._dirty_records) == 0
    assert bob._is_dirty is False
    assert thingamajig._is_dirty is False

    bob_copy = conn2.t.Widget.Get("id=?", bob.id)
    thing_copy = conn2.t.OtherWidget.Get("id=?", thingamajig.id)

    assert bob_copy.age == 22
    assert thingamajig.weight == 12
    assert thingamajig.name == "New Hotness 2.0"
    # Intentional and intended, change tracking is only available when explicitly tracking
    assert thingamajig.height == 5


def test_dcdb__cast_to_database_AND_cast_from_database___column_as_enum(connection):
    import enum

    class EnumFlags(enum.Enum):
        OFF = 0
        ON = 1

        @classmethod
        def To(cls, value, _):
            return value.name

        @classmethod
        def From(cls, value, _):
            return cls.__members__[value]

    @dataclass()
    class Test:
        enum_column: EnumFlags = EnumFlags.OFF

    connection.bind(Test)
    default_record = connection.t.Test()
    off_record = connection.t.Test(enum_column=EnumFlags.OFF)
    on_record = connection.t.Test(enum_column=EnumFlags.ON)

    assert off_record.enum_column == EnumFlags.OFF
    assert on_record.enum_column == EnumFlags.ON


def test_RelationshipFields_DOT_unordered_list___works_and_replaces_AutoList(connection):
    import enum

    class ChildTableStatus(enum.IntEnum):
        PENDING = 0
        COMPLETE = 1

        @classmethod
        def To(cls, value, _):
            return value.value if not isinstance(value, int) else value

        @classmethod
        def From(cls, value, _):
            return cls(int(value))

    @dataclass()
    class Parent:
        name: str

        def _relationships(self, instance):

            instance.Steves = dcdb.RelationshipFields.unordered_list("Children", "parent_id", where="name LIKE 'Steve%'")

            instance.Pending = dcdb.RelationshipFields.unordered_list("Children", "parent_id"
                                                         , where=f"status = {ChildTableStatus.PENDING.value}"
                                                         , add_set=("status", ChildTableStatus.PENDING.value,)
                                                         , remove_set=("status", ChildTableStatus.PENDING.value,)
                                                         )

            instance.Complete = dcdb.RelationshipFields.unordered_list("Children", "parent_id"
                                                          , where=f"status = {ChildTableStatus.COMPLETE.value}"
                                                          , add_set=("status", ChildTableStatus.COMPLETE.value,)
                                                          , remove_set=("status", ChildTableStatus.PENDING.value,)
                                                          )

    @dataclass()
    class Children:
        name: str
        parent_id: int = None
        status: ChildTableStatus = ChildTableStatus.PENDING

    connection.binds(Parent, Children)

    steve, stjr, joe, joejr, \
    carl, alice, smith = \
        connection.t.Children.Insert_many(
            {"name": "Steve"},
            {"name": "Steve JR."},
            {"name": "Joe"},
            {"name": "Joe Jr."},
            {"name": "Carl", "status": ChildTableStatus.COMPLETE},
            {"name": "Alice", "status": ChildTableStatus.COMPLETE},
            {"name": "Smith", "status": ChildTableStatus.COMPLETE},
        )

    bob = connection.t.Parent(name="Bob")

    completed_count = connection.t.Children.Count("status=?", ChildTableStatus.COMPLETE.value)
    pending_count = connection.t.Children.Count("status=?", ChildTableStatus.PENDING.value)
    assert completed_count == 3
    assert pending_count == 4

    result = bob.Pending
    assert len(result) == 0
    bob.Pending += [steve, stjr, joe, joejr]
    assert len(bob.Pending) == 4
    bob.Pending.pop(joe)
    assert len(bob.Pending) == 3
    assert len(bob.Steves) == 2

    bob.Complete += [carl, alice, smith]
    assert len(bob.Complete) == 3
    bob.Complete.remove(alice)
    assert len(bob.Complete) == 2


def test_RelationshipFields_DOT_unordered_list__dotted_relations(connection):
    @dataclass()
    class Boss:
        name: str
        def _relationships(self, instance):
            instance.employees = dcdb.RelationshipFields.unordered_list("Employee", "boss_id")

    @dataclass()
    class Employee:
        name: str
        age: int
        boss_id: int = None

    connection.bind_scan(locals())

    boss = connection.t.Boss(name="Bill")

    emp1 = connection.t.Employee(name="Joe", age=12)
    emp2 = connection.t.Employee(name="Al", age=22)
    emp3 = connection.t.Employee(name="Jane", age=30)
    emp4 = connection.t.Employee(name="Friday", age=28)

    boss.employees.insert(emp1)
    boss.employees.insert(emp2)
    boss.employees.insert(emp3)

    assert len(boss.employees) == 3
    assert connection.t.Employee.Count() == 4


def test_Transformers_AND_Datetime_dot_date():
    actual_date = dt.date(2018, 3, 5)
    actual_date_str = "2018-03-05"
    assert dcdb.Transformers.To(actual_date, dt.date) == actual_date_str
    assert dcdb.Transformers.To(actual_date_str, dt.date) == actual_date_str

    assert dcdb.Transformers.From(actual_date_str, dt.date) == actual_date


def test_Transformers__handles_decimal():
    import decimal

    dcdb.Transformers.Set(decimal.Decimal, lambda v, t: str(v), lambda v, t: t(v))

    actual = 1.0
    to_val = "1.0"

    assert dcdb


def main():
    import sys
    import pathlib as pl

    my_path = pl.Path(__file__).resolve().parent
    sys.path.append(str(my_path))

    pytest.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
