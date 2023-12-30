
import argparse
  

class HyperParametersQNet():
    def parse(self):
        parser = argparse.ArgumentParser()
        # training parameters
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--optim', type=str, default='SGD')

        # dataset params
        parser.add_argument('--train-set', type=str, default='subset_train_4', help='Train set')
        # model parameters
        parser.add_argument('--arch', type=str, default='resnet18')
        # db loader parameters
        parser.add_argument('--num-workers', type=int, default=2, help='Num workers for the pytorch dataloader')
        parser.add_argument('--port',  default='2222', type=str,  help='Port to run')
        
        self.args = vars(parser.parse_args())

        assert self.args['optim'] in {'Adam', 'SGD'}, 'Invalid optimizer'
        assert self.args['arch'] in {'resnet50', 'resnet18', 'small', 'resnet101'}


    def __getitem__(self, key):
        return self.args[key]

    def __str__(self):
        return str(self.args)


