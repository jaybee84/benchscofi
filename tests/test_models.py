import unittest
from glob import glob
from subprocess import call

if __name__ == '__main__':
    # initialize the test suite
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    # add tests to the test suite
    model_folder='../src/benchscofi/'
    model_lst=[x.split(model_folder)[-1].split('.py')[0] for x in glob(model_folder+'*.py') if (x!=model_folder+'__init__.py')]
    for model in model_lst:
        call("sed s/XXXXXX/"+model+"/g TemplateTest.py > Test"+model+".py", shell=True)
    suite.addTests(loader.discover("./", pattern="Test*.py"))
    call("rm -f "+" ".join(glob("./Test*.py")), shell=True)
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)