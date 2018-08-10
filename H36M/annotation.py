class Annotation(object):
    IMG = 'image'  # string, image name
    S = 'S'  # np.array(float32), 3d position in camera space
    CENTER = 'center'  # np.array(float32), center in image space
    PART = 'part'  # np.array(float32), joint position in image space
    SCALE = 'scale'  # np.array(float32), proposition of human body in image
    ZIDX = 'zind'  # np.array(float32), voxel z index, need to clip [1, 64]
    SUBJECT = 'subject'  # string, subject number "S#"
    ACTION = 'action'  # string, action name
    FRAME = 'frame'  # string, frame number of video
    CAM = 'camera'  # string, camera serial number
    INTRINSIC = 'intrinsic'  # np.array(float32), fckp 2,2,3,2 total 9 elem

    @staticmethod
    def get_annot_list():
        return [Annotation.IMG,
                Annotation.S,
                Annotation.CENTER,
                Annotation.PART,
                Annotation.SCALE,
                Annotation.ZIDX,
                Annotation.SUBJECT,
                Annotation.ACTION,
                Annotation.FRAME,
                Annotation.CAM,
                Annotation.INTRINSIC,
                ]
