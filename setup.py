from setuptools import setup, find_packages

setup(name='multiagent',
      version='0.0.1',
      description='Multi-Agent Goal-Driven Communication Environment',
      # url='https://github.com/openai/multiagent-public',
      author='Debajit Chakraborty',
      author_email='c.debajit15@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
