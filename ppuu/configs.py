import argparse
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Iterable, Union, NewType, Any

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


@dataclass
class ConfigBase:
    """Base class that should handle parsing from command line,
    json, dicts.
    """

    @classmethod
    def parse_from_command_line(cls):
        result = DataclassArgParser(cls).parse_args_into_dataclasses()
        if len(result) > 1:
            raise RuntimeError(
                f"The following arguments were not recognized: {result[1:]}"
            )
        return result[0]

    @classmethod
    def parse_from_dict(cls, inputs):
        return DataclassArgParser(cls)._populate_dataclass_from_dict(
            cls, inputs.copy()
        )


@dataclass
class ModelConfig(ConfigBase):
    model_type: str = field(default="vanilla")
    checkpoint: Union[str, None] = field(default=None)


DATASET_PATHS_MAPPING = {
    "full": (
        "/misc/vlgscratch4/LecunGroup/nvidia-collab/traffic-data_offroad/state-action-cost/data_i80_v0/"  # noqa: E501
    ),
    "50": (
        "/misc/vlgscratch4/LecunGroup/nvidia-collab/vlad/traffic-data_offroad_50_test_train_same/state-action-cost/data_i80_v0/"  # noqa: E501
    ),
}


@dataclass
class TrainingConfig(ConfigBase):
    """The class that holds configurations common to all training
    scripts.  Does not contain model configurations.
    """

    learning_rate: float = field(default=0.0001)
    n_epochs: int = field(default=101)
    epoch_size: int = field(default=500)
    batch_size: int = field(default=6)
    validation_size: int = field(default=25)
    dataset: str = field(default="full")
    data_shift: bool = field(default=False)
    random_actions: bool = field(default=False)
    seed: int = field(default=42)
    output_dir: str = field(default=None)
    experiment_name: str = field(default="train_mpur")
    slurm: bool = field(default=False)
    run_eval: bool = field(default=False)
    debug: bool = field(default=False)
    fast_dev_run: bool = field(default=False)
    freeze_encoder: bool = field(default=False)
    mixout_p: float = field(default=None)
    validation_eval: bool = field(default=True)
    noise_augmentation_std: float = field(default=0.07)
    noise_augmentation_p: float = field(default=0.5)
    gpus: int = field(default=1)
    num_nodes: int = field(default=1)
    distributed_backend: str = field(default='ddp')

    def __post_init__(self):
        self.set_dataset(self.dataset)

    def set_dataset(self, dataset):
        self.dataset = dataset
        if self.dataset in DATASET_PATHS_MAPPING:
            self.dataset = DATASET_PATHS_MAPPING[self.dataset]


@dataclass
class DataConfig(ConfigBase):
    """This holds all configurations pertaining to data loading,
    mainly contains paths.
    """

    pass

    def __post_init__(self):
        pass


class DataclassArgParser(argparse.ArgumentParser):
    """A class for handling dataclasses and argument parsing.
    Closely based on Hugging Face's HfArgumentParser class,
    extended to support recursive dataclasses.
    """

    def __init__(
        self,
        dataclass_types: Union[DataClassType, Iterable[DataClassType]],
        **kwargs,
    ):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will
                "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular
                way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for f in dataclasses.fields(dtype):
            field_name = f"--{f.name}"
            kwargs = f.metadata.copy()
            typestring = str(f.type)
            for x in (int, float, str):
                if typestring == f"typing.Union[{x.__name__}, NoneType]":
                    f.type = x
            if isinstance(f.type, type) and issubclass(f.type, Enum):
                kwargs["choices"] = list(f.type)
                kwargs["type"] = f.type
                if f.default is not dataclasses.MISSING:
                    kwargs["default"] = f.default
            elif f.type is bool:
                kwargs["action"] = (
                    "store_false" if f.default is True else "store_true"
                )
                if f.default is True:
                    field_name = f"--no-{f.name}"
                    kwargs["dest"] = f.name
            elif dataclasses.is_dataclass(f.type):
                self._add_dataclass_arguments(f.type)
            else:
                kwargs["type"] = f.type
                if f.default is not dataclasses.MISSING:
                    kwargs["default"] = f.default
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(self, args=None,) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass
        types.  This relies on argparse's `ArgumentParser.parse_known_args`.
        See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args
        Args:
            args:
                List of strings to parse. The default is taken from sys.argv.
                (same as argparse.ArgumentParser)
        Returns:
            Tuple consisting of:
                - the dataclass instances in the same order as they
                  were passed to the initializer.abspath
                - if applicable, an additional namespace for more
                  (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings.
                  (same as argparse.ArgumentParser.parse_known_args)
        """
        namespace, unknown = self.parse_known_args(args=args)
        outputs = []

        for dtype in self.dataclass_types:
            outputs.append(self._populate_dataclass(dtype, namespace))
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if len(unknown) > 0:
            outputs.append(unknown)
        return outputs

    @staticmethod
    def _populate_dataclass(
        dtype: DataClassType, namespace: argparse.Namespace
    ):
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in vars(namespace).items() if k in keys}
        for k in keys:
            delattr(namespace, k)
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            inputs[k] = DataclassArgParser._populate_dataclass(s, namespace)
        obj = dtype(**inputs)
        return obj

    @staticmethod
    def _populate_dataclass_from_dict(dtype: DataClassType, d: dict):
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in d.items() if k in keys}
        for k in keys:
            if k in d:
                del d[k]
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            inputs[k] = DataclassArgParser._populate_dataclass_from_dict(s, d)
        obj = dtype(**inputs)
        return obj
