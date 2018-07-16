import argparse


def iters_per_min(t_train, n_iters=1001):
    return n_iters / (t_train / 60)


def megavox_per_sec(t_inf, n_vox=3072*3072*200):
    return n_vox / t_inf / 1e6


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='train or inference')
    parser.add_argument('time', type=float)

    args = parser.parse_args()
    if args.mode == 'train':
        print(iters_per_min(args.time))
    elif args.mode == 'inference':
        print(megavox_per_sec(args.time))
    else:
        raise TypeError("Invalid argument %s" % args.mode)
