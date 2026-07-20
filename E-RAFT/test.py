import numpy as np
import torch as th
from torchvision import utils
from utils.helper_functions import *
import utils.visualization as visualization
import utils.filename_templates as TEMPLATES
import utils.helper_functions as helper
import utils.logger as logger
from utils import image_utils
import os
import matplotlib.pyplot as plt

class Test(object):
    """
    Test class

    """

    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        self.downsample = False # Downsampling for Rebuttal
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.additional_args = additional_args
        if config['cuda'] and not torch.cuda.is_available():
            print('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        if save_path is None:
            self.save_path = helper.create_save_path(config['save_dir'].lower(),
                                           config['name'].lower())
        else:
            self.save_path=save_path
        if logger is None:
            self.logger = logger.Logger(self.save_path)
        else:
            self.logger = test_logger
        if isinstance(self.additional_args, dict) and 'name_mapping_test' in self.additional_args.keys():
            visu_add_args = {'name_mapping' : self.additional_args['name_mapping_test']}
        else:
            visu_add_args = None
        self.visualizer = visualizer(data_loader, self.save_path, additional_args=visu_add_args)

    def summary(self):
        self.logger.write_line("====================================== TEST SUMMARY ======================================", True)
        self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
        self.logger.write_line("Tester:\t\t" + self.__class__.__name__, True)
        self.logger.write_line("Test Set:\t" + self.data_loader.dataset.__class__.__name__, True)
        self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
        self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
        self.logger.write_line("==========================================================================================", True)

    def run_network(self, epoch):
        raise NotImplementedError

    def move_batch_to_cuda(self, batch):
        raise NotImplementedError

    def visualize_sample(self, batch):
        self.visualizer(batch)

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch, batch_idx, None)

    def get_estimation_and_target(self, batch):
        # Returns the estimation and target of the current batch
        raise NotImplementedError

    def flow_map2event(self, flow_map, event_volume_old):
        # Convert the flow map to event volume
        # print(flow_map.shape, event_volume_old.shape)
        event_flow_map = []
        event_flow_map_nonzero = np.empty((4,0), int)
        im_grid = np.meshgrid(np.arange(flow_map.shape[2]), np.arange(flow_map.shape[1]), sparse=False)
        im_grid = np.asarray(im_grid)

        for i in range(event_volume_old.shape[0]):
            event_flow_map.append(event_volume_old[i]*flow_map)
            # print(event_flow_map[-1].shape,im_grid.shape)
            flow_w_coor = np.concatenate((event_flow_map[-1],im_grid),axis=0).reshape(4,-1)
            flow_w_coor = flow_w_coor[:,(flow_w_coor[0] != 0) & (flow_w_coor[1] != 0)]
            event_flow_map_nonzero = np.hstack((event_flow_map_nonzero,flow_w_coor))
            # print(event_flow_map_nonzero.shape)

        # event_flow_map_nonzero = np.array(event_flow_map_nonzero)
        # print(event_flow_map_nonzero.shape)


        # return np.array(event_flow_map)
        return event_flow_map_nonzero
    def _test(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        os.mkdir(self.save_path + '/flow/')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                # Move Data to GPU
                if next(self.model.parameters()).is_cuda:
                    batch = self.move_batch_to_cuda(batch)
                # Network Forward Pass
                self.run_network(batch)
                print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))

                # Get Estimation and Target for MVSEC
                if self.config['name'] == 'mvsec_20Hz':
                    flow_est, flow_gt = self.get_estimation_and_target(batch)
                    flow_convert = flow_est[0].cpu().numpy()

                    # # old event volume
                    # event_volume_old = batch[0]['event_volume_old'].cpu().numpy()
                    # event_volume_old_sum = event_volume_old.sum(axis=1)
                    # event_volume_old_sum = np.where(event_volume_old_sum == 0, 0, 1)
                    # event_volume_old = np.where(event_volume_old == 0, 0, 1)
                    # event_flow_convert = self.flow_map2event(flow_convert, event_volume_old[0])
                    # # print(event_flow_convert[14], event_volume_old[0][14], flow_convert)
                    # flow_convert = flow_convert * event_volume_old_sum

                    # new event volume
                    event_volume_new = batch[0]['event_volume_new'].cpu().numpy()
                    event_volume_new_sum = event_volume_new.sum(axis=1)
                    event_volume_new_sum = np.where(event_volume_new_sum == 0, 0, 1)
                    event_volume_new = np.where(event_volume_new == 0, 0, 1)
                    event_flow_convert = self.flow_map2event(flow_convert, event_volume_new[0])
                    # print(event_flow_convert[14], event_volume_old[0][14], flow_convert)
                    flow_convert = flow_convert * event_volume_new_sum
                    # plt.imsave(self.save_path + '/flow/' + str(batch_idx + 1) + '.jpg', flow_convert[0])

                    # print(batch[0]['flow_est'].shape)
                    batch[0]['flow_est'][0] = th.from_numpy(flow_convert).to(self.gpu)
                    # print(event_volume_old.shape, flow_convert.shape)
                    # print(batch[0]['idx'])
                    # print(flow_est[0].)
                    # np.save(
                    #     self.save_path+ '/flow/' + str(
                    #         batch_idx + 1) + '.npy', flow_est[0].cpu().numpy())

                    # # MVSEC
                    # np.save(
                    #     self.save_path+ '/flow/' + str(
                    #         batch_idx + 1) + '.npy', event_flow_convert)
                    # EVIMO
                    np.save(
                        self.save_path + '/flow/' + str(
                            batch_idx + 1) + '.npy', flow_convert)

                    # time reversed
                    # np.save(
                    #     self.save_path + '/flow/' + str(
                    #         1396 - batch_idx + 1) + '.npy', event_flow_convert)
                elif self.config['name'] == 'dsec_warm_start':
                    flow_est = batch[0]['flow_est']
                    flow_convert = flow_est[0].cpu().numpy()
                    event_volume_new = batch[0]['event_volume_new'].cpu().numpy()
                    event_volume_new_sum = event_volume_new.sum(axis=1)
                    event_volume_new_sum = np.where(event_volume_new_sum == 0, 0, 1)
                    event_volume_new = np.where(event_volume_new == 0, 0, 1)
                    event_flow_convert = self.flow_map2event(flow_convert, event_volume_new[0])
                    # print(event_flow_convert[14], event_volume_old[0][14], flow_convert)
                    flow_convert = flow_convert * event_volume_new_sum
                    print(flow_convert.shape)


                    batch[0]['flow_est'][0] = th.from_numpy(flow_convert).to(self.gpu)

                    np.save(
                        self.save_path + '/flow/' + str(
                            batch[0]['file_index'].cpu().numpy()[0].astype(np.uint8)) + '.npy', event_flow_convert)



                # Visualize
                if hasattr(batch, 'keys') and 'loader_idx' in batch.keys() \
                        or (isinstance(batch,list) and hasattr(batch[0], 'keys') and 'loader_idx' in batch[0].keys()):
                    self.visualize_sample(batch)
                else:
                    # DSEC Special Snowflake
                    self.visualize_sample_dsec(batch, batch_idx)
                    #print('Not Visualizing')

        # Log Generation
        log = {}

        return log

class TestRaftEvents(Test):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch.keys():
                return batch['flow_est'].cpu().data, (batch['flow'].cpu().data, batch['gt_valid_mask'].cpu().data)
            return batch['flow_est'].cpu().data, batch['flow'].cpu().data
        else:
            f_est = batch['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def run_network(self, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        _, batch['flow_list'] = self.model(image1=im1,
                                           image2=im2)
        batch['flow_est'] = batch['flow_list'][-1]

class TestRaftEventsWarm(Test):
    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        super(TestRaftEventsWarm, self).__init__(model, config,
                                                 data_loader, visualizer, test_logger, save_path,
                                                 additional_args=additional_args)
        self.subtype = config['subtype'].lower()
        print('Tester Subtype: {}'.format(self.subtype))
        self.net_init = None # Hidden state of the refinement GRU
        self.flow_init = None
        self.idx_prev = None
        self.init_print = False
        assert self.data_loader.batch_size == 1, 'Batch size for recurrent testing must be 1'

    def move_batch_to_cuda(self, batch):
        return move_list_to_cuda(batch, self.gpu)

    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch[-1].keys():
                return batch[-1]['flow_est'].cpu().data, (batch[-1]['flow'].cpu().data, batch[-1]['gt_valid_mask'].cpu().data)
            return batch[-1]['flow_est'].cpu().data, batch[-1]['flow'].cpu().data
        else:
            f_est = batch[-1]['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch[-1]['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch[-1].keys():
                f_mask = torch.nn.functional.interpolate(batch[-1]['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def visualize_sample(self, batch):
        self.visualizer(batch[-1])

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch[-1], batch_idx, None)

    def check_states(self, batch):
        # 0th case: there is a flag in the batch that tells us to reset the state (DSEC)
        if 'new_sequence' in batch[0].keys():
            if batch[0]['new_sequence'].item() == 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
        else:
            # During Validation, reset state if a new scene starts (index jump)
            if self.idx_prev is not None and batch[0]['idx'].item() - self.idx_prev != 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
            self.idx_prev = batch[0]['idx'].item()

    def run_network(self, batch):
        self.check_states(batch)
        for l in range(len(batch)):
            # Run Recurrent Network for this sample

            if not self.downsample:
                im1 = batch[l]['event_volume_old']
                im2 = batch[l]['event_volume_new']
            else:
                im1 = torch.nn.functional.interpolate(batch[l]['event_volume_old'], scale_factor=0.5)
                im2 = torch.nn.functional.interpolate(batch[l]['event_volume_new'], scale_factor=0.5)
            flow_low_res, batch[l]['flow_list'] = self.model(image1=im1,
                                                                image2=im2,
                                                                flow_init=self.flow_init)

        batch[l]['flow_est'] = batch[l]['flow_list'][-1]
        self.flow_init = image_utils.forward_interpolate_pytorch(flow_low_res)
        batch[l]['flow_init'] = self.flow_init
