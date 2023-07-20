import unittest
from glob import glob
from subprocess import call
import sys

if __name__ == '__main__':
    # initialize the test suite
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    # add tests to the test suite
    model_folder='../src/benchscofi/'
    all_models=[x.split(model_folder)[-1].split('.py')[0] for x in glob(model_folder+'*.py') if (x!=model_folder+'__init__.py')]
    model_lst=all_models if (len(sys.argv)<=1) else [sys.argv[1]]
    dataset='' if (len(sys.argv)<=2) else sys.argv[2]
    assert len(model_lst)>1 or (model_lst[0] in all_models)
    for model in model_lst:
        call("sed s/XXXXXX/"+model+"/g TemplateTest.py > Test"+model+".py", shell=True)
        if (len(dataset)>0):
            call("sed -i s/YYYYYYYYYYY/"+dataset+"/g Test"+model+".py", shell=True)
    suite.addTests(loader.discover("./", pattern="Test*.py"))
    call("rm -f "+" ".join(glob("./Test*.py")), shell=True)
    # initialize a runner, pass it your suite and run it
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(suite)