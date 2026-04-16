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
python attack_main.py --attack mifgsm --regularization KL --weight 100
python attack_main.py --attack mifgsm --regularization KL --weight 10
python attack_main.py --attack mifgsm --regularization KL --weight 1
python attack_main.py --attack mifgsm --regularization KL --weight 0.1
python attack_main.py --attack mifgsm --regularization KL --weight 0.001
