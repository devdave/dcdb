.. DCDB documentation master file, created by
   sphinx-quickstart on Thu Dec  6 09:41:23 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DCDB's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. contents::
   :local:


Define a table/model
====================

.. code-block:: python

   @dataclass
   class Foo:
      a_number: int
      text: str

   connection = dcdb.DBConnection("file_path/mydb.sqlite3")
   connection.bind(Foo)

Creates the sqlite table Foo with columns `a_number` and `text`.


Create and retrieve a record
============================

.. code-block:: python

   @dataclass
   class Foo:
      a_number: int
      text: str

   connection = dcdb.DBConnection("file_path/mydb.sqlite3")
   connection.bind(Foo)

   record = connection.tables.Foo(a_number=123, text="Hello world")
   # record is automatically saved to database on creation
   same_record = connect.tables.Foo.Select("a_number=?", 123)
   assert same_record.id == record.id # True
   assert same_record.text == "Hello World" # True


DCDB module
===========

.. automodule:: dcdb.dcdb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
