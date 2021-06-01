from setuptools import find_packages
from cx_Freeze import setup, Executable


options = {
    'build_exe': {
        'includes': [
            'cx_Logging', 'idna',
        ],
        'packages': [
            'asyncio', 'flask', 'jinja2', 'dash', 'plotly'
        ],
        'excludes': ['tkinter']
    }
}

executables = [
    Executable('quality_app.py',
               base='console',
               targetName='quality_app.exe')
]

setup(
    name='quality_app',
    packages=find_packages(),
    version='1.0.0',
    description='quality_app',
    executables=executables,
    options=options
)