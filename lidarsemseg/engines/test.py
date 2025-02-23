"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.utils.data

import open3d as o3d

from lidarsemseg.utils.visualization import save_txt_and_pcd

from .defaults import create_ddp_model
import lidarsemseg.utils.comm as comm
from lidarsemseg.dataprocessin import build_dataset, collate_fn
from lidarsemseg.network import build_model
from lidarsemseg.utils.logger import get_root_logger
from lidarsemseg.utils.registry import Registry
from lidarsemseg.utils.misc import (
    AverageMeter,
    intersection_and_union,
    make_dirs,
)


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):

    def test(self):

        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

                # Define colors for each class (RGB format)
        class_colors = {
            0: [0.0, 0.8, 0.0],     # Grass - Green
            1: [0.0, 0.5, 0.0],     # Low Vegetation - Dark Green
            2: [0.8, 0.6, 0.4],     # Rough Trail - Brown
            3: [0.9, 0.9, 0.0],     # Smooth Trail - Yellow
            4: [0.0, 0.3, 0.0],     # High Vegetation - Forest Green
            5: [0.7, 0.7, 0.7],     # Obstacle - Gray
        }

        # Clear existing predictions
        save_path = os.path.join(self.cfg.save_path, "result")
        pcd_save_path = os.path.join(save_path, "colored_pcds")
        if comm.is_main_process():
            if os.path.exists(save_path):
                import shutil
                shutil.rmtree(save_path)
            make_dirs(save_path)
            make_dirs(pcd_save_path)  # Create directory for colored point clouds
            make_dirs(os.path.join(save_path, "submit"))  # Create submit directory

        # Ensure all processes wait for directory creation
        comm.synchronize()

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)

        # create submit folder only on main process
        if comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))

        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]
            
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            
            original_coords = []
            

            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                    
                original_coords.append(input_dict['coord'])
                # print(original_coords)

                for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                    
                with torch.no_grad():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                logger.info(
                    "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                        idx + 1,
                        len(self.test_loader),
                        data_name=data_name,
                        batch_idx=i,
                        batch_num=len(fragment_list),
                        )
                    )

                # original_coords = torch.cat(original_coords, dim=0)
                    
            pred = pred.max(1)[1].data.cpu().numpy()

                # points = original_coords.cpu().numpy()
                # colors = np.zeros_like(points)  # Shape: (N, 3) for RGB values

                # # Get predicted classes
                # pred_classes = pred

                # # Assign colors based on predicted classes
                # for class_id, color in class_colors.items():
                #         mask = pred_classes == class_id  # Shape: (N,)
                #         colors[mask] = color  # Assign RGB values to matching points

                # # Save both formats
                # txt_path = os.path.join(pcd_save_path, f"{data_name}_colored.txt")
                # pcd_path = os.path.join(pcd_save_path, f"{data_name}_colored.pcd")
                # save_txt_and_pcd(points, colors, txt_path, pcd_path)

            print(len(pred))
            if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
            np.save(pred_save_path, pred)
            
            # Save predictions
            np.savetxt(
                os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                pred.reshape([-1, 1]),
                fmt="%d",
            )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
