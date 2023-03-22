from argparse import Namespace

import torch

one_shot_messages = set()


def config_check(args):
    pass


class ConfigManager(Namespace):
    def __init__(self, args=None, **kwargs):
        """
        The ConfigManager is a subclass of argparse.Namespace and based on parameter dict and count the call-frequency of each parameter
        :param args: A parameter dict
        :param kwargs: Same param as Namespce
        """
        if not args:
            args = {}
        super().__init__(**kwargs)

        if isinstance(args, Namespace):
            self.args = vars(args)
            self.args_call_count = {arg: 0 for arg in vars(args)}
        else:
            self.args = args
            self.args_call_count = {arg: 0 for arg in args}

    def __getattribute__(self, arg_name):
        if arg_name == "args" or arg_name == "args_call_count":
            return super().__getattribute__(arg_name)
        try:
            value = super().__getattribute__("args")[arg_name]
            args_call_count = super().__getattribute__("args_call_count")
            args_call_count[arg_name] += 1
            super().__setattr__("args_call_count", args_call_count)
            return value

        except Exception as e:
            return super().__getattribute__(arg_name)

    def __setattr__(self, arg_name, value):
        if arg_name == "args" or arg_name == "args_call_count":
            super().__setattr__(arg_name, value)
            return
        try:
            args = super().__getattribute__("args")
            args[arg_name] = value
            super().__setattr__("args", args)
            args_call_count = super().__getattribute__("args_call_count")

            if arg_name in args_call_count:
                # args_call_count[arg_name] += 1
                super().__setattr__("args_call_count", args_call_count)

            else:
                args_call_count[arg_name] = 0
                super().__setattr__("args_call_count", args_call_count)

        except Exception as e:
            super().__setattr__(arg_name, value)

        config_check(args)
