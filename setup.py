from setuptools import setup,find_packages
from typing import List

hypen_e='-e .'
def get_reqs(file_path:str)-> List:
    '''this function returns the list of requirements'''
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if hypen_e in requirements:
            requirements.remove(hypen_e)


    return requirements




setup(
name='mlprojects',
version='0.1',
author='VarshanthG',
author_email='Varshanthg2030@gmail.com',
packages=find_packages(),
install_requires=get_reqs('requirements.txt')
)
