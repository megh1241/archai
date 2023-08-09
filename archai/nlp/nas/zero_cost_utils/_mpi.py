import logging
import os
import abc

class MPIExperimentSetup(metaclass=abc.ABCMeta):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def setup_gpus(self, ds_colocated=False):
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.threads = True
        mpi4py.rc.thread_level = "multiple"
        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init_thread()

    def evaluator_method_kwargs(self):
        t = {"comm": self.workcomm}
        return t

    def trainer_method_kwargs(self):
        return {}
