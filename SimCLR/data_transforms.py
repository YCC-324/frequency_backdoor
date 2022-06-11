from torchvision import transforms

cifar10_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
    transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
    transforms.ToTensor(),   #PIL->Tensor
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

cifar10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

cifar100_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
    transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
    transforms.ToTensor(),   #PIL->Tensor
    transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])

cifar100_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])

svhn_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
    transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
    transforms.ToTensor(),   #PIL->Tensor
    transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])

svhn_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614])])

stl10_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(96),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
    transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
    transforms.ToTensor(),   #PIL->Tensor
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

stl10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

mnist_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])])

gtsrb_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
    transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
    transforms.ToTensor(),   #PIL->Tensor
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

gtsrb_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

imagenet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),    #随机改变亮度、对比度、饱和度
    transforms.RandomGrayscale(p=0.2),   #依概率将图片转换成灰度图
    transforms.ToTensor(),   #PIL->Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

imagenet_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])