import os
import re
import yaml
import collections


class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append("{}:".format(key))
                child_ret_str = value.__repr__().split("\n")
                for item in child_ret_str:
                    ret_str.append("    " + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append("{}:".format(key))
                    for item in value:
                        # Treat as AttrDict above.
                        child_ret_str = item.__repr__().split("\n")
                        for item in child_ret_str:
                            ret_str.append("    " + item)
                else:
                    ret_str.append("{}: {}".format(key, value))
            else:
                ret_str.append("{}: {}".format(key, value))
        return "\n".join(ret_str)


class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, verbose=False):
        super(Config, self).__init__()
        # Update with given configurations.
        assert os.path.exists(filename), "File {} not exist.".format(filename)
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        try:
            with open(filename, "r") as f:
                cfg_dict = yaml.load(f, Loader=loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        recursive_update(self, cfg_dict)


def yml_tuple_constructor(loader, node):
    # this little parse is really just for what I needed, feel free to change it!
    def parse_tup_el(el):
        # try to convert into int or float else keep the string
        if el.isdigit():
            return int(el)
        try:
            return float(el)
        except ValueError:
            return el

    value = loader.construct_scalar(node)
    # remove the ( ) from the string
    tup_elements = value[1:-1].split(",")
    # remove the last element if the tuple was written as (x,b,)
    if tup_elements[-1] == "":
        tup_elements.pop(-1)
    tup = tuple(map(parse_tup_el, tup_elements))
    return tup


def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d
