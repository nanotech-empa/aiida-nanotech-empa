from setuptools import setup, find_packages
import json

if __name__ == '__main__':
    # Provide static information in setup.json
    # such that it can be discovered automatically
    with open('setup.json', 'r', encoding='utf-8') as info:
        kwargs = json.load(info)
    with open('README.md', encoding='utf-8') as readmef:
        readme = readmef.read()
    setup(
        packages=find_packages(),
        # this doesn't work when placed in setup.json (something to do with str type)
        package_data={
            "": ["*"],
        },
        long_description=readme,
        long_description_content_type='text/markdown',
        **kwargs)
