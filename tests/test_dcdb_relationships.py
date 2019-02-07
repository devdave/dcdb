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
    box.contents.create(name="Hammer", quantity=2)
    box.contents.create(name="Measuring tape", quantity=1)
    box.contents.create(name="Bandaids", quantity=25)

    many_widgets = box.contents.where("quantity > ?", 1 ) # type: dcdb.DBCursorProxy
    alot_widgets = box.contents.where("quantity >= ?", 25) # type: dcdb.DBCursorProxy

    assert len(list(many_widgets)) == 3
    assert len(list(alot_widgets.fetchall())) == 2

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
    assert bandaids.quantity == 25
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
    assert bits in storagebox.things

    assert tape not in toolbox.things
    assert tape in storagebox.things

    assert connection.t.Thing.Count() == 5
    assert len(toolbox.things) == 3
    assert len(storagebox.things) == 3

    assert toolbox.things['Nails'].quantity == 150
    assert storagebox.things['Bits'].quantity == 12

    copy_screwdriver = toolbox.things['Screwdriver']
    copy_screwdriver.quantity = 1
    copy_screwdriver.save()
    assert storagebox.things['Screwdriver'].quantity == 1


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



