import argparse

def parameter_parser():
   
    parser = argparse.ArgumentParser(description = "Run .")
    parser.add_argument('--root', type=str, default='../data')

    # 控制攻击方法、攻击类型和攻击率
    parser.add_argument('--attack', type=str, default='random_add', choices=['none', 'random_add','random_flip','random_remove','cdelm','flipm', 'meta_attack','add','del','gflipm','gdelm'])
    parser.add_argument('--ptb_rate', type=float, default=0.20, help='pertubation rate')
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=2, choices=[1, 2, 3], help='noisy level')

    parser.add_argument("--epochs",
                        type = int,
                        default = 200,
	                    help = "Number of training epochs. Default is 200.")
    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                    help = "Random seed. Default is 42.")
    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                    help = "Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                    help = "Learning rate. Default is 0.01.")
    parser.add_argument("--dataset",
                        nargs="?",
                        default='cora',
                        help="The name of data set. Default is cora.")
    parser.add_argument("--community-size",  #限制了找到的社区大小最多为30
                        type=int,
                        default=30,
                        help="The size of final community. Default is 30.")

    parser.add_argument("--train-ratio",
                        type=float,
                        default=0.02,
                        help="Test data ratio. Default is 0.02.")

    parser.add_argument("--subgraph-size",  #限制在这个返回内查询
                       type=int,
                       default=400,
                       help="The size of subgraphs. Default is 400. when you try on facebook,it should be smaller")
    parser.add_argument("--layers",
                        type=int,
                        default=[16],
                        nargs='+',
                        help="The size of hidden layers. Default is [16].")
    parser.add_argument("--seed-cnt",
                        type=int,
                        default=500,
                        help="The number of random seeds. Default is 20."
                        )
    parser.add_argument("--iteration",
                        type=bool,
                        default=False,
                        help="Whether to start iteration. Default is False."
                        )
    parser.add_argument("--upsize",
                        type=int,
                        default=20,
                        help="Maximum number of node can be found per iteration. Default is 20."
                        )
    parser.add_argument("--possize",
                        type=int,
                        default=1,
                        help="Incremental train node pairs per iteration. Default is 1."
                        )
    parser.add_argument("--round",
                        type=int,
                        default=10,
                        help="The number of iteration rounds. Default is 10."
                        )
    #parser.set_defaults(layers = [16, 16, 16])
    #parser.set_defaults(layers=[hidden]*layer_len)
    return parser.parse_args()
