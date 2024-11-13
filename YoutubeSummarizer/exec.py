import os
import subprocess
import calendar

def get_mondays(year):
    mondays = []
    for month in range(1,13):
        cal = calendar.monthcalendar(year, month)
        for week in cal:
            if week[calendar.MONDAY] != 0:
                mondays.append(f"{year}-{month:02d}-{week[calendar.MONDAY]:02d}")
    return mondays

year = 2023
mondays = get_mondays(year)


## activation
venv_path = '.\ys\Scripts' ## update your python path
python_path = os.path.join(venv_path, 'python')

for monday in mondays:
    print(monday)
    p = subprocess.run([python_path, 'main.py', '-t', monday])