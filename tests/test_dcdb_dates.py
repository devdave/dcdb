

def test_Transformers_AND_Datetime_dot_date():
    actual_date = dt.date(2018, 3, 5)
    actual_date_str = "2018-03-05"
    assert dcdb.Transformers.To(actual_date, dt.date) == actual_date_str
    assert dcdb.Transformers.To(actual_date_str, dt.date) == actual_date_str

    assert dcdb.Transformers.From(actual_date_str, dt.date) == actual_date



def test_DefaultLocalTime___allow_default_time(connection):

    import datetime

    @dataclass()
    class ThatTime:
        created_on: dcdb.DefaultLocalTime() = dcdb.DefaultLocalTime.ColDef

    connection.bind(ThatTime)


    now = datetime.datetime.now()
    created_now = connection.t.ThatTime()

    assert created_now.created_on.year == now.year
    assert created_now.created_on.month == now.month
    assert created_now.created_on.day == now.day
    assert created_now.created_on.hour == now.hour
    assert now.minute+1 > created_now.created_on.minute > now.minute - 1
