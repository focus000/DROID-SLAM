import torch
from droid_net import DroidNet
from depth_video import DepthVideo
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict


class DroidUnseq:
    def __init__(
        self,
        args,
        tstamps: torch.Tensor,
        images: torch.Tensor,
        poses: torch.Tensor,
        calibs: torch.Tensor,
    ):
        super(DroidUnseq, self).__init__()
        self.load_weights(args.weights)
        self.args = args

        self.init_video(tstamps, images, poses, calibs)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """context features"""
        net, inp = self.net.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """features for correlation volume"""
        return self.net.fnet(image).squeeze(0)

    def load_weights(self, weights):
        """load trained model weights"""

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict(
            [(k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()]
        )

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def init_video(
        self,
        tstamps: torch.Tensor,
        images: torch.Tensor,
        poses: torch.Tensor,
        calibs: torch.Tensor,
        # device: str,
    ):
        assert tstamps.shape[0] == images.shape[0] == calibs.shape[0]

        h, w = images.shape[1], images.shape[2]
        buffer = images.shape[0]

        # normalize images
        inputs = images[None, :, [2, 1, 0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)
        net, inp = self.__context_encoder(inputs[:, [0]])

        self.video = DepthVideo([h, w], buffer, stereo=self.args.stereo)

        self.video.tstamp = tstamps
        self.video.images = images
        self.video.poses = poses
        self.video.intrinsics = calibs / 8.0
        self.video.fmaps = gmap
        self.video.nets = net
        self.video.inps = inp
        self.video.counter.value += tstamps.shape[0]

    def terminate(self, stream=None):
        """terminate the visualization process, return poses [t, q]"""

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()
