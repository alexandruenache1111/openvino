Quantization Aware Training with NNCF, using PyTorch framework
==============================================================

This notebook is based on `ImageNet training in
PyTorch <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__.

The goal of this notebook is to demonstrate how to use the Neural
Network Compression Framework
`NNCF <https://github.com/openvinotoolkit/nncf>`__ 8-bit quantization to
optimize a PyTorch model for inference with OpenVINO Toolkit. The
optimization process contains the following steps:

-  Transforming the original ``FP32`` model to ``INT8``
-  Using fine-tuning to improve the accuracy.
-  Exporting optimized and original models to OpenVINO IR
-  Measuring and comparing the performance of models.

For more advanced usage, refer to these
`examples <https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

This tutorial uses the ResNet-18 model with the Tiny ImageNet-200
dataset. ResNet-18 is the version of ResNet models that contains the
fewest layers (18). Tiny ImageNet-200 is a subset of the larger ImageNet
dataset with smaller images. The dataset will be downloaded in the
notebook. Using the smaller model and dataset will speed up training and
download time. To see other ResNet models, visit `PyTorch
hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__.

   **NOTE**: This notebook requires a C++ compiler for compiling PyTorch
   custom operations for quantization. For Windows we recommend to
   install Visual Studio with C++ support, you can find instruction
   `here <https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170>`__.
   For MacOS ``xcode-select --install`` command installs many developer
   tools, including C++. For Linux you can install gcc with your
   distribution’s package manager.


**Table of contents:**


-  `Imports and Settings <#imports-and-settings>`__
-  `Pre-train Floating-Point Model <#pre-train-floating-point-model>`__

   -  `Train Function <#train-function>`__
   -  `Validate Function <#validate-function>`__
   -  `Helpers <#helpers>`__
   -  `Get a Pre-trained FP32 Model <#get-a-pre-trained-fp32-model>`__

-  `Create and Initialize
   Quantization <#create-and-initialize-quantization>`__
-  `Fine-tune the Compressed Model <#fine-tune-the-compressed-model>`__
-  `Export INT8 Model to OpenVINO
   IR <#export-int8-model-to-openvino-ir>`__
-  `Benchmark Model Performance by Computing Inference
   Time <#benchmark-model-performance-by-computing-inference-time>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu  "openvino>=2024.0.0" "torch" "torchvision" "tqdm"
    %pip install -q "nncf>=2.9.0"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    

Imports and Settings
--------------------



On Windows, add the required C++ directories to the system PATH.

Import NNCF and all auxiliary packages from your Python code. Set a name
for the model, and the image width and height that will be used for the
network. Also define paths where PyTorch and OpenVINO IR versions of the
models will be stored.

   **NOTE**: All NNCF logging messages below ERROR level (INFO and
   WARNING) are disabled to simplify the tutorial. For production use,
   it is recommended to enable logging by removing
   ``set_log_level(logging.ERROR)``.

.. code:: ipython3

    import time
    import warnings  # To disable warnings on export model
    import zipfile
    from pathlib import Path
    
    import torch
    
    import torch.nn as nn
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    import openvino as ov
    from torch.jit import TracerWarning
    
    # Fetch `notebook_utils` module
    import requests
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
    
        open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, device_widget
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    DATA_DIR = Path("data")
    BASE_MODEL_NAME = "resnet18"
    image_size = 64
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Paths where PyTorch and OpenVINO IR models will be stored.
    fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
    fp32_ir_path = fp32_pth_path.with_suffix(".xml")
    int8_ir_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".xml")
    
    # It is possible to train FP32 model from scratch, but it might be slow. Therefore, the pre-trained weights are downloaded by default.
    pretrained_on_tiny_imagenet = True
    fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
    if not (MODEL_DIR / fp32_pth_path).exists():
        download_file(fp32_pth_url, directory=MODEL_DIR, filename=fp32_pth_path.name)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("pytorch-quantization-aware-training.ipynb")


.. parsed-literal::

    Using cuda device
    


.. parsed-literal::

    model/resnet18_fp32.pth:   0%|          | 0.00/43.1M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/home/dlyakhov/Projects/openvino_notebooks/notebooks/pytorch-quantization-aware-training/model/resnet18_fp32.pth')



Download Tiny ImageNet dataset

-  100k images of shape 3x64x64
-  200 different classes: snake, spider, cat, truck, grasshopper, gull,
   etc.

.. code:: ipython3

    def download_tiny_imagenet_200(
        data_dir: Path,
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        tarname="tiny-imagenet-200.zip",
    ):
        archive_path = data_dir / tarname
        download_file(url, directory=data_dir, filename=tarname)
        zip_ref = zipfile.ZipFile(archive_path, "r")
        zip_ref.extractall(path=data_dir)
        zip_ref.close()
    
    
    def prepare_tiny_imagenet_200(dataset_dir: Path):
        # Format validation set the same way as train set is formatted.
        val_data_dir = dataset_dir / "val"
        val_annotations_file = val_data_dir / "val_annotations.txt"
        with open(val_annotations_file, "r") as f:
            val_annotation_data = map(lambda line: line.split("\t")[:2], f.readlines())
        val_images_dir = val_data_dir / "images"
        for image_filename, image_label in val_annotation_data:
            from_image_filepath = val_images_dir / image_filename
            to_image_dir = val_data_dir / image_label
            if not to_image_dir.exists():
                to_image_dir.mkdir()
            to_image_filepath = to_image_dir / image_filename
            from_image_filepath.rename(to_image_filepath)
        val_annotations_file.unlink()
        val_images_dir.rmdir()
    
    
    DATASET_DIR = DATA_DIR / "tiny-imagenet-200"
    if not DATASET_DIR.exists():
        download_tiny_imagenet_200(DATA_DIR)
        prepare_tiny_imagenet_200(DATASET_DIR)
        print(f"Successfully downloaded and prepared dataset at: {DATASET_DIR}")



.. parsed-literal::

    data/tiny-imagenet-200.zip:   0%|          | 0.00/237M [00:00<?, ?B/s]


.. parsed-literal::

    Successfully downloaded and prepared dataset at: data/tiny-imagenet-200
    

Pre-train Floating-Point Model
------------------------------



Using NNCF for model compression assumes that a pre-trained model and a
training pipeline are already in use.

This tutorial demonstrates one possible training pipeline: a ResNet-18
model pre-trained on 1000 classes from ImageNet is fine-tuned with 200
classes from Tiny-ImageNet.

Subsequently, the training and validation functions will be reused as is
for quantization-aware training.

Train Function
~~~~~~~~~~~~~~



.. code:: ipython3

    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix="Epoch:[{}]".format(epoch),
        )
    
        # Switch to train mode.
        model.train()
    
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
    
            # Compute output.
            output = model(images)
            loss = criterion(output, target)
    
            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
            # Compute gradient and do opt step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()
    
            print_frequency = 50
            if i % print_frequency == 0:
                progress.display(i)

Validate Function
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")
    
        # Switch to evaluate mode.
        model.eval()
    
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(device)
    
                # Compute output.
                output = model(images)
                loss = criterion(output, target)
    
                # Measure accuracy and record loss.
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
    
                # Measure elapsed time.
                batch_time.update(time.time() - end)
                end = time.time()
    
                print_frequency = 10
                if i % print_frequency == 0:
                    progress.display(i)
    
            print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
        return top1.avg

Helpers
~~~~~~~



.. code:: ipython3

    class AverageMeter(object):
        """Computes and stores the average and current value"""
    
        def __init__(self, name, fmt=":f"):
            self.name = name
            self.fmt = fmt
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)
    
    
    class ProgressMeter(object):
        def __init__(self, num_batches, meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix
    
        def display(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print("\t".join(entries))
    
        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = "{:" + str(num_digits) + "d}"
            return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    
    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
    
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

Get a Pre-trained FP32 Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



А pre-trained floating-point model is a prerequisite for quantization.
It can be obtained by tuning from scratch with the code below. However,
this usually takes a lot of time. Therefore, this code has already been
run and received good enough weights after 4 epochs (for the sake of
simplicity, tuning was not done until the best accuracy). By default,
this notebook just loads these weights without launching training. To
train the model yourself on a model pre-trained on ImageNet, set
``pretrained_on_tiny_imagenet = False`` in the Imports and Settings
section at the top of this notebook.

.. code:: ipython3

    num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    init_lr = 1e-4
    batch_size = 128
    epochs = 4
    
    model = models.resnet18(pretrained=not pretrained_on_tiny_imagenet)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    
    # Data loading code.
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)


.. parsed-literal::

    /home/dlyakhov/env/tmp/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /home/dlyakhov/env/tmp/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
      warnings.warn(msg)
    

.. code:: ipython3

    if pretrained_on_tiny_imagenet:
        #
        # ** WARNING: The `torch.load` functionality uses Python's pickling module that
        # may be used to perform arbitrary code execution during unpickling. Only load data that you
        # trust.
        #
        checkpoint = torch.load(str(fp32_pth_path), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        acc1_fp32 = checkpoint["acc1"]
    else:
        best_acc1 = 0
        # Training loop.
        for epoch in range(0, epochs):
            # Run a single training epoch.
            train(train_loader, model, criterion, optimizer, epoch)
    
            # Evaluate on validation set.
            acc1 = validate(val_loader, model, criterion)
    
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
    
            if is_best:
                checkpoint = {"state_dict": model.state_dict(), "acc1": acc1}
                torch.save(checkpoint, fp32_pth_path)
        acc1_fp32 = best_acc1
    
    print(f"Accuracy of FP32 model: {acc1_fp32:.3f}")


.. parsed-literal::

    Accuracy of FP32 model: 55.520
    

Export the ``FP32`` model to OpenVINO™ Intermediate Representation, to
benchmark it in comparison with the ``INT8`` model.

.. code:: ipython3

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    ov_model = ov.convert_model(model, example_input=dummy_input, input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
    print(f"FP32 model was exported to {fp32_ir_path}.")


.. parsed-literal::

    FP32 model was exported to model/resnet18_fp32.xml.
    

Create and Initialize Quantization
----------------------------------



NNCF enables compression-aware training by integrating into regular
training pipelines. The framework is designed so that modifications to
your original training code are minor. Quantization requires only 2
modifications.

1. Create a quantization data loader with batch size equal to one and
   wrap it by the ``nncf.Dataset``, specifying a transformation function
   which prepares input data to fit into model during quantization. In
   our case, to pick input tensor from pair (input tensor and label).

.. code:: ipython3

    import nncf
    
    
    def transform_fn(data_item):
        return data_item[0]
    
    
    # Creating separate dataloader with batch size = 1
    # as dataloaders with batches > 1 is not supported yet.
    quantization_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    quantization_dataset = nncf.Dataset(quantization_loader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino
    

2. Run ``nncf.quantize`` for Getting an Optimized Model.

``nncf.quantize`` function accepts model and prepared quantization
dataset for performing basic quantization. Optionally, additional
parameters like ``subset_size``, ``preset``, ``ignored_scope`` can be
provided to improve quantization result if applicable. More details
about supported parameters can be found on this
`page <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html#tune-quantization-parameters>`__

.. code:: ipython3

    quantized_model = nncf.quantize(model, quantization_dataset)


.. parsed-literal::

    2024-01-17 15:43:43.543878: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-17 15:43:43.579576: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-01-17 15:43:44.170538: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    

.. parsed-literal::

    WARNING:nncf:NNCF provides best results with torch==2.1.0, while current torch version is 1.13.0+cu117. If you encounter issues, consider switching to torch==2.1.0
    


.. parsed-literal::

    Output()






    







    


.. parsed-literal::

    INFO:nncf:Compiling and loading torch extension: quantized_functions_cuda...
    INFO:nncf:Finished loading torch extension: quantized_functions_cuda
    


.. parsed-literal::

    Output()






    







    


Evaluate the new model on the validation set after initialization of
quantization. The accuracy should be close to the accuracy of the
floating-point ``FP32`` model for a simple case like the one being
demonstrated here.

.. code:: ipython3

    acc1 = validate(val_loader, quantized_model, criterion)
    print(f"Accuracy of initialized INT8 model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.110 (0.110)	Loss 0.992 (0.992)	Acc@1 78.12 (78.12)	Acc@5 89.06 (89.06)
    Test: [10/79]	Time 0.069 (0.074)	Loss 1.990 (1.623)	Acc@1 44.53 (60.37)	Acc@5 79.69 (83.95)
    Test: [20/79]	Time 0.068 (0.072)	Loss 1.814 (1.704)	Acc@1 60.16 (58.26)	Acc@5 80.47 (82.63)
    Test: [30/79]	Time 0.068 (0.071)	Loss 2.284 (1.794)	Acc@1 52.34 (56.75)	Acc@5 67.97 (80.90)
    Test: [40/79]	Time 0.070 (0.072)	Loss 1.618 (1.831)	Acc@1 61.72 (55.64)	Acc@5 82.03 (80.37)
    Test: [50/79]	Time 0.069 (0.071)	Loss 1.951 (1.832)	Acc@1 57.81 (55.70)	Acc@5 75.00 (80.06)
    Test: [60/79]	Time 0.070 (0.071)	Loss 1.795 (1.855)	Acc@1 56.25 (55.28)	Acc@5 84.38 (79.75)
    Test: [70/79]	Time 0.069 (0.071)	Loss 2.359 (1.888)	Acc@1 47.66 (54.79)	Acc@5 74.22 (79.08)
     * Acc@1 55.130 Acc@5 79.680
    Accuracy of initialized INT8 model: 55.130
    

Fine-tune the Compressed Model
------------------------------



At this step, a regular fine-tuning process is applied to further
improve quantized model accuracy. Normally, several epochs of tuning are
required with a small learning rate, the same that is usually used at
the end of the training of the original model. No other changes in the
training pipeline are required. Here is a simple example.

.. code:: ipython3

    compression_lr = init_lr / 10
    optimizer = torch.optim.Adam(quantized_model.parameters(), lr=compression_lr)
    
    # Train for one epoch with NNCF.
    train(train_loader, quantized_model, criterion, optimizer, epoch=0)
    
    # Evaluate on validation set after Quantization-Aware Training (QAT case).
    acc1_int8 = validate(val_loader, quantized_model, criterion)
    
    print(f"Accuracy of tuned INT8 model: {acc1_int8:.3f}")
    print(f"Accuracy drop of tuned INT8 model over pre-trained FP32 model: {acc1_fp32 - acc1_int8:.3f}")


.. parsed-literal::

    Epoch:[0][  0/782]	Time 0.284 (0.284)	Loss 0.876 (0.876)	Acc@1 78.12 (78.12)	Acc@5 92.97 (92.97)
    Epoch:[0][ 50/782]	Time 0.112 (0.116)	Loss 0.796 (0.808)	Acc@1 80.47 (79.96)	Acc@5 94.53 (94.27)
    Epoch:[0][100/782]	Time 0.111 (0.114)	Loss 0.785 (0.788)	Acc@1 82.81 (80.52)	Acc@5 92.19 (94.56)
    Epoch:[0][150/782]	Time 0.114 (0.113)	Loss 0.653 (0.785)	Acc@1 84.38 (80.69)	Acc@5 95.31 (94.45)
    Epoch:[0][200/782]	Time 0.109 (0.113)	Loss 0.804 (0.780)	Acc@1 80.47 (80.92)	Acc@5 94.53 (94.45)
    Epoch:[0][250/782]	Time 0.111 (0.113)	Loss 0.756 (0.777)	Acc@1 83.59 (80.98)	Acc@5 94.53 (94.47)
    Epoch:[0][300/782]	Time 0.112 (0.112)	Loss 0.665 (0.772)	Acc@1 82.03 (81.07)	Acc@5 96.88 (94.53)
    Epoch:[0][350/782]	Time 0.115 (0.112)	Loss 0.661 (0.767)	Acc@1 82.81 (81.14)	Acc@5 97.66 (94.57)
    Epoch:[0][400/782]	Time 0.111 (0.113)	Loss 0.661 (0.764)	Acc@1 78.91 (81.24)	Acc@5 96.09 (94.60)
    Epoch:[0][450/782]	Time 0.119 (0.113)	Loss 0.904 (0.762)	Acc@1 79.69 (81.27)	Acc@5 89.06 (94.60)
    Epoch:[0][500/782]	Time 0.113 (0.113)	Loss 0.609 (0.757)	Acc@1 84.38 (81.46)	Acc@5 96.88 (94.62)
    Epoch:[0][550/782]	Time 0.112 (0.113)	Loss 0.833 (0.753)	Acc@1 76.56 (81.59)	Acc@5 95.31 (94.69)
    Epoch:[0][600/782]	Time 0.112 (0.113)	Loss 0.768 (0.751)	Acc@1 82.81 (81.63)	Acc@5 95.31 (94.69)
    Epoch:[0][650/782]	Time 0.112 (0.113)	Loss 0.750 (0.751)	Acc@1 82.81 (81.61)	Acc@5 93.75 (94.71)
    Epoch:[0][700/782]	Time 0.110 (0.113)	Loss 0.654 (0.749)	Acc@1 84.38 (81.62)	Acc@5 96.09 (94.71)
    Epoch:[0][750/782]	Time 0.110 (0.113)	Loss 0.575 (0.748)	Acc@1 86.72 (81.67)	Acc@5 97.66 (94.73)
    Test: [ 0/79]	Time 0.070 (0.070)	Loss 1.028 (1.028)	Acc@1 78.91 (78.91)	Acc@5 86.72 (86.72)
    Test: [10/79]	Time 0.070 (0.070)	Loss 1.827 (1.514)	Acc@1 46.88 (63.35)	Acc@5 79.69 (84.02)
    Test: [20/79]	Time 0.073 (0.070)	Loss 1.628 (1.594)	Acc@1 64.06 (60.97)	Acc@5 82.03 (83.78)
    Test: [30/79]	Time 0.069 (0.070)	Loss 2.061 (1.688)	Acc@1 57.03 (59.25)	Acc@5 71.88 (82.26)
    Test: [40/79]	Time 0.070 (0.070)	Loss 1.495 (1.738)	Acc@1 66.41 (57.93)	Acc@5 85.16 (81.59)
    Test: [50/79]	Time 0.069 (0.070)	Loss 1.863 (1.741)	Acc@1 58.59 (57.83)	Acc@5 76.56 (81.31)
    Test: [60/79]	Time 0.069 (0.070)	Loss 1.571 (1.779)	Acc@1 65.62 (57.21)	Acc@5 84.38 (80.74)
    Test: [70/79]	Time 0.069 (0.070)	Loss 2.505 (1.809)	Acc@1 46.09 (56.78)	Acc@5 75.00 (80.22)
     * Acc@1 57.200 Acc@5 80.880
    Accuracy of tuned INT8 model: 57.200
    Accuracy drop of tuned INT8 model over pre-trained FP32 model: -1.680
    

Export INT8 Model to OpenVINO IR
--------------------------------



.. code:: ipython3

    if not int8_ir_path.exists():
        warnings.filterwarnings("ignore", category=TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # Export INT8 model to OpenVINO™ IR
        ov_model = ov.convert_model(quantized_model, example_input=dummy_input, input=[1, 3, image_size, image_size])
        ov.save_model(ov_model, int8_ir_path)
        print(f"INT8 model exported to {int8_ir_path}.")


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.
    INT8 Omodel exported to model/resnet18_int8.xml.
    

Benchmark Model Performance by Computing Inference Time
-------------------------------------------------------



Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
- inference performance measurement tool in OpenVINO. By default,
Benchmark Tool runs inference for 60 seconds in asynchronous mode on
CPU. It returns inference speed as latency (milliseconds per image) and
throughput (frames per second) values.

   **NOTE**: This notebook runs ``benchmark_app`` for 15 seconds to give
   a quick indication of performance. For more accurate performance, it
   is recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications. Run
   ``benchmark_app -m model.xml -d CPU`` to benchmark async inference on
   CPU for one minute. Change CPU to GPU to benchmark on GPU. Run
   ``benchmark_app --help`` to see an overview of all command-line
   options.

.. code:: ipython3

    device = device_widget()
    
    device

.. code:: ipython3

    def parse_benchmark_output(benchmark_output):
        parsed_output = [line for line in benchmark_output if "FPS" in line]
        print(*parsed_output, sep="\n")
    
    
    print("Benchmark FP32 model (IR)")
    benchmark_output = ! benchmark_app -m $fp32_ir_path -d $device.value -api async -t 15
    parse_benchmark_output(benchmark_output)
    
    print("Benchmark INT8 model (IR)")
    benchmark_output = ! benchmark_app -m $int8_ir_path -d $device.value -api async -t 15
    parse_benchmark_output(benchmark_output)


.. parsed-literal::

    Benchmark FP32 model (IR)
    [ INFO ] Throughput:   3755.92 FPS
    Benchmark INT8 model (IR)
    [ INFO ] Throughput:   15141.53 FPS
    

Show Device Information for reference.

.. code:: ipython3

    import openvino.properties as props
    
    
    core = ov.Core()
    core.get_property(device.value, props.device.full_name)




.. parsed-literal::

    'Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz'


