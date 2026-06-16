# python optimized_attack_result.py --model densenet121
# python optimized_attack_result.py --model resnext50
# python optimized_attack_result.py --model vgg19bn
# python optimized_attack_result.py --model incres_v2
# python optimized_attack_result.py --model inc_v3
# python optimized_attack_result.py --model inc_v4
# python optimized_attack_result.py --model resnet101
# python optimized_attack_result.py --model resnet152

# python optimized_attack_result_incremental.py --attack mifgsm --regularization CE
# python optimized_attack_result_incremental.py --attack mifgsm --regularization KL

# python attack_main.py --attack mifgsm --regularization CE
# python attack_main.py --attack mifgsm --regularization MSE
# python attack_main.py --attack mifgsm --regularization KL --weight 0.01
# python attack_main.py --attack mifgsm --regularization KL --weight 1000
# python attack_main.py --attack mifgsm --regularization KL --weight 100
# python attack_main.py --attack mifgsm --regularization KL --weight 10
# python attack_main.py --attack mifgsm --regularization KL --weight 1
# python attack_main.py --attack mifgsm --regularization KL --weight 0.1
# python attack_main.py --attack mifgsm --regularization KL --weight 0.001

python attack_main.py --attack difgsm --regularization CE
python attack_main.py --attack difgsm --regularization MSE
python attack_main.py --attack difgsm --regularization KL --weight 0.01
python attack_main.py --attack difgsm --regularization KL --weight 1000
python attack_main.py --attack difgsm --regularization KL --weight 100
python attack_main.py --attack difgsm --regularization KL --weight 10
python attack_main.py --attack difgsm --regularization KL --weight 1
python attack_main.py --attack difgsm --regularization KL --weight 0.1
python attack_main.py --attack difgsm --regularization KL --weight 0.001

python attack_main.py --attack tifgsm --regularization CE
python attack_main.py --attack tifgsm --regularization MSE
python attack_main.py --attack tifgsm --regularization KL --weight 0.01
python attack_main.py --attack tifgsm --regularization KL --weight 1000
python attack_main.py --attack tifgsm --regularization KL --weight 100
python attack_main.py --attack tifgsm --regularization KL --weight 10
python attack_main.py --attack tifgsm --regularization KL --weight 1
python attack_main.py --attack tifgsm --regularization KL --weight 0.1
python attack_main.py --attack tifgsm --regularization KL --weight 0.001

python attack_main.py --attack nifgsm --regularization CE
python attack_main.py --attack nifgsm --regularization MSE
python attack_main.py --attack nifgsm --regularization KL --weight 0.01
python attack_main.py --attack nifgsm --regularization KL --weight 1000
python attack_main.py --attack nifgsm --regularization KL --weight 100
python attack_main.py --attack nifgsm --regularization KL --weight 10
python attack_main.py --attack nifgsm --regularization KL --weight 1
python attack_main.py --attack nifgsm --regularization KL --weight 0.1
python attack_main.py --attack nifgsm --regularization KL --weight 0.001

python attack_main.py --attack sinifgsm --regularization CE
python attack_main.py --attack sinifgsm --regularization MSE
python attack_main.py --attack sinifgsm --regularization KL --weight 0.01
python attack_main.py --attack sinifgsm --regularization KL --weight 1000
python attack_main.py --attack sinifgsm --regularization KL --weight 100
python attack_main.py --attack sinifgsm --regularization KL --weight 10
python attack_main.py --attack sinifgsm --regularization KL --weight 1
python attack_main.py --attack sinifgsm --regularization KL --weight 0.1
python attack_main.py --attack sinifgsm --regularization KL --weight 0.001

python attack_main.py --attack vmifgsm --regularization CE
python attack_main.py --attack vmifgsm --regularization MSE
python attack_main.py --attack vmifgsm --regularization KL --weight 0.01
python attack_main.py --attack vmifgsm --regularization KL --weight 1000
python attack_main.py --attack vmifgsm --regularization KL --weight 100
python attack_main.py --attack vmifgsm --regularization KL --weight 10
python attack_main.py --attack vmifgsm --regularization KL --weight 1
python attack_main.py --attack vmifgsm --regularization KL --weight 0.1
python attack_main.py --attack vmifgsm --regularization KL --weight 0.001

python attack_main.py --attack vnifgsm --regularization CE
python attack_main.py --attack vnifgsm --regularization MSE
python attack_main.py --attack vnifgsm --regularization KL --weight 0.01
python attack_main.py --attack vnifgsm --regularization KL --weight 1000
python attack_main.py --attack vnifgsm --regularization KL --weight 100
python attack_main.py --attack vnifgsm --regularization KL --weight 10
python attack_main.py --attack vnifgsm --regularization KL --weight 1
python attack_main.py --attack vnifgsm --regularization KL --weight 0.1
python attack_main.py --attack vnifgsm --regularization KL --weight 0.001

# python attack_main.py --model resnet50
# python attack_main.py --model densenet121
# python attack_main.py --model resnext50
# python attack_main.py --model vgg19bn
# python attack_main.py --model incres_v2
# python attack_main.py --model inc_v3
# python attack_main.py --model inc_v4
# python attack_main.py --model resnet101
# python attack_main.py --model resnet152
# python attack_main.py --model adv_inception_v3
# python attack_main.py --model ens_adv_inception_resnet_v2
# python attack_main.py --model visformer_small
# python attack_main.py --model vit_b
# python attack_main.py --model swin_b
# python attack_main.py --model pit_b
# python attack_main.py --model mobilenet

# python attack_main.py --attack difgsm --model resnet50
# python attack_main.py --attack difgsm --model densenet121
# python attack_main.py --attack difgsm --model resnext50
# python attack_main.py --attack difgsm --model vgg19bn
# python attack_main.py --attack difgsm --model incres_v2
# python attack_main.py --attack difgsm --model inc_v3
# python attack_main.py --attack difgsm --model inc_v4
# python attack_main.py --attack difgsm --model resnet101
# python attack_main.py --attack difgsm --model resnet152
# python attack_main.py --attack difgsm --model adv_inception_v3
# python attack_main.py --attack difgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack difgsm --model visformer_small
# python attack_main.py --attack difgsm --model vit_b
# python attack_main.py --attack difgsm --model swin_b
# python attack_main.py --attack difgsm --model pit_b
# python attack_main.py --attack difgsm --model mobilenet

# python attack_main.py --attack tifgsm --model resnet50
# python attack_main.py --attack tifgsm --model densenet121
# python attack_main.py --attack tifgsm --model resnext50
# python attack_main.py --attack tifgsm --model vgg19bn
# python attack_main.py --attack tifgsm --model incres_v2
# python attack_main.py --attack tifgsm --model inc_v3
# python attack_main.py --attack tifgsm --model inc_v4
# python attack_main.py --attack tifgsm --model resnet101
# python attack_main.py --attack tifgsm --model resnet152
# python attack_main.py --attack tifgsm --model adv_inception_v3
# python attack_main.py --attack tifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack tifgsm --model visformer_small
# python attack_main.py --attack tifgsm --model vit_b
# python attack_main.py --attack tifgsm --model swin_b
# python attack_main.py --attack tifgsm --model pit_b
# python attack_main.py --attack tifgsm --model mobilenet

# python attack_main.py --attack nifgsm --model resnet50
# python attack_main.py --attack nifgsm --model densenet121
# python attack_main.py --attack nifgsm --model resnext50
# python attack_main.py --attack nifgsm --model vgg19bn
# python attack_main.py --attack nifgsm --model incres_v2
# python attack_main.py --attack nifgsm --model inc_v3
# python attack_main.py --attack nifgsm --model inc_v4
# python attack_main.py --attack nifgsm --model resnet101
# python attack_main.py --attack nifgsm --model resnet152
# python attack_main.py --attack nifgsm --model adv_inception_v3
# python attack_main.py --attack nifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack nifgsm --model visformer_small
# python attack_main.py --attack nifgsm --model vit_b
# python attack_main.py --attack nifgsm --model swin_b
# python attack_main.py --attack nifgsm --model pit_b
# python attack_main.py --attack nifgsm --model mobilenet

# python attack_main.py --attack sinifgsm --model resnet50
# python attack_main.py --attack sinifgsm --model densenet121
# python attack_main.py --attack sinifgsm --model resnext50
# python attack_main.py --attack sinifgsm --model vgg19bn
# python attack_main.py --attack sinifgsm --model incres_v2
# python attack_main.py --attack sinifgsm --model inc_v3
# python attack_main.py --attack sinifgsm --model inc_v4
# python attack_main.py --attack sinifgsm --model resnet101
# python attack_main.py --attack sinifgsm --model resnet152
# python attack_main.py --attack sinifgsm --model adv_inception_v3
# python attack_main.py --attack sinifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack sinifgsm --model visformer_small
# python attack_main.py --attack sinifgsm --model vit_b
# python attack_main.py --attack sinifgsm --model swin_b
# python attack_main.py --attack sinifgsm --model pit_b
# python attack_main.py --attack sinifgsm --model mobilenet

# python attack_main.py --attack vmifgsm --model resnet50
# python attack_main.py --attack vmifgsm --model densenet121
# python attack_main.py --attack vmifgsm --model resnext50
# python attack_main.py --attack vmifgsm --model vgg19bn
# python attack_main.py --attack vmifgsm --model incres_v2
# python attack_main.py --attack vmifgsm --model inc_v3
# python attack_main.py --attack vmifgsm --model inc_v4
# python attack_main.py --attack vmifgsm --model resnet101
# python attack_main.py --attack vmifgsm --model resnet152
# python attack_main.py --attack vmifgsm --model adv_inception_v3
# python attack_main.py --attack vmifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack vmifgsm --model visformer_small
# python attack_main.py --attack vmifgsm --model vit_b
# python attack_main.py --attack vmifgsm --model swin_b
# python attack_main.py --attack vmifgsm --model pit_b
# python attack_main.py --attack vmifgsm --model mobilenet

# python attack_main.py --attack vnifgsm --model resnet50
# python attack_main.py --attack vnifgsm --model densenet121
# python attack_main.py --attack vnifgsm --model resnext50
# python attack_main.py --attack vnifgsm --model vgg19bn
# python attack_main.py --attack vnifgsm --model incres_v2
# python attack_main.py --attack vnifgsm --model inc_v3
# python attack_main.py --attack vnifgsm --model inc_v4
# python attack_main.py --attack vnifgsm --model resnet101
# python attack_main.py --attack vnifgsm --model resnet152
# python attack_main.py --attack vnifgsm --model adv_inception_v3
# python attack_main.py --attack vnifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack vnifgsm --model visformer_small
# python attack_main.py --attack vnifgsm --model vit_b
# python attack_main.py --attack vnifgsm --model swin_b
# python attack_main.py --attack vnifgsm --model pit_b
# python attack_main.py --attack vnifgsm --model mobilenet

# python attack_main.py --attack bsrmifgsm --model resnet50
# python attack_main.py --attack bsrmifgsm --model densenet121
# python attack_main.py --attack bsrmifgsm --model resnext50
# python attack_main.py --attack bsrmifgsm --model vgg19bn
# python attack_main.py --attack bsrmifgsm --model incres_v2
# python attack_main.py --attack bsrmifgsm --model inc_v3
# python attack_main.py --attack bsrmifgsm --model inc_v4
# python attack_main.py --attack bsrmifgsm --model resnet101
# python attack_main.py --attack bsrmifgsm --model resnet152
# python attack_main.py --attack bsrmifgsm --model adv_inception_v3
# python attack_main.py --attack bsrmifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack bsrmifgsm --model visformer_small
# python attack_main.py --attack bsrmifgsm --model vit_b
# python attack_main.py --attack bsrmifgsm --model swin_b
# python attack_main.py --attack bsrmifgsm --model pit_b
# python attack_main.py --attack bsrmifgsm --model mobilenet

# python attack_main.py --attack ggsmifgsm --model resnet50
# python attack_main.py --attack ggsmifgsm --model densenet121
# python attack_main.py --attack ggsmifgsm --model resnext50
# python attack_main.py --attack ggsmifgsm --model vgg19bn
# python attack_main.py --attack ggsmifgsm --model incres_v2
# python attack_main.py --attack ggsmifgsm --model inc_v3
# python attack_main.py --attack ggsmifgsm --model inc_v4
# python attack_main.py --attack ggsmifgsm --model resnet101
# python attack_main.py --attack ggsmifgsm --model resnet152
# python attack_main.py --attack ggsmifgsm --model adv_inception_v3
# python attack_main.py --attack ggsmifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack ggsmifgsm --model visformer_small
# python attack_main.py --attack ggsmifgsm --model vit_b
# python attack_main.py --attack ggsmifgsm --model swin_b
# python attack_main.py --attack ggsmifgsm --model pit_b
# python attack_main.py --attack ggsmifgsm --model mobilenet

# python attack_main.py --attack sidmifgsm --model resnet50
# python attack_main.py --attack sidmifgsm --model densenet121
# python attack_main.py --attack sidmifgsm --model resnext50
# python attack_main.py --attack sidmifgsm --model vgg19bn
# python attack_main.py --attack sidmifgsm --model incres_v2
# python attack_main.py --attack sidmifgsm --model inc_v3
# python attack_main.py --attack sidmifgsm --model inc_v4
# python attack_main.py --attack sidmifgsm --model resnet101
# python attack_main.py --attack sidmifgsm --model resnet152
# python attack_main.py --attack sidmifgsm --model adv_inception_v3
# python attack_main.py --attack sidmifgsm --model ens_adv_inception_resnet_v2
# python attack_main.py --attack sidmifgsm --model visformer_small
# python attack_main.py --attack sidmifgsm --model vit_b
# python attack_main.py --attack sidmifgsm --model swin_b
# python attack_main.py --attack sidmifgsm --model pit_b
# python attack_main.py --attack sidmifgsm --model mobilenet
