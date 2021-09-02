import numpy as np

class Evaluator:
    def __init__(self):
        self.cnt = 0
        self.x = np.arange(start=-50, stop=50, step=1)
        """
        You can initialize your model here
        """
        self.result = []

    def iou2d_caculate(self, input_source: dict, input_target: dict):
        """
        计算两个矩形框的交并比。
        :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
        :param rec2: (x0,y0,x1,y1)
        :return: 交并比IOU.
        """
        rec1 = (input_source['box2d']['xmin'],
                input_source['box2d']['ymin'],
                input_source['box2d']['xmax'],
                input_source['box2d']['ymax'])
        rec2 = (input_target['box2d']['xmin'],
                input_target['box2d']['ymin'],
                input_target['box2d']['xmax'],
                input_target['box2d']['ymax'])
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        # 两矩形无相交区域的情况
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0
        # 两矩形有相交区域的情况
        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross / (S1 + S2 - S_cross)


    def evaluate_one_data(self, input_source: dict, input_target: dict) -> dict:

        if "CLASSIFICATION" in input_source.keys() and "CLASSIFICATION" in input_target.keys():
            if input_source["CLASSIFICATION"]['attributes']['traffic'] == input_target["CLASSIFICATION"]['attributes']['traffic']:
                res = 1
            else:
                res = -1
            ratio = np.tan(res)
            y = self.x * ratio

            dict_ret = {
                'overall': {
                    'float_file_metric': self.cnt,
                    'customized_iou': res,
                    'curve_file_metric': {
                        'x': self.x.tolist(),
                        'y': y.tolist()
                    }
                }
            }
            self.result = self.result + [res]

        elif "BOX2D" in input_source.keys() and "BOX2D" in input_target.keys():
            paired_iou_list = []
            source_list = input_source['BOX2D']
            target_list = input_target['BOX2D']
            for s in source_list:
                for t in target_list:
                    if s['category'] == t['category']:
                        iou = self.iou2d_caculate(s, t)
                        if iou > 0.5:
                            paired_iou_list = paired_iou_list + [iou]
            if len(paired_iou_list) != 0:
                iou_mean = float(np.array(paired_iou_list).mean())
            else:
                iou_mean = 0

            ratio = np.tan(iou_mean)
            y = self.x * ratio

            dict_ret = {
                    'overall': {
                        'float_file_metric': self.cnt,
                        'customized_iou': iou_mean,
                        'curve_file_metric': {
                            'x': self.x.tolist(),
                            'y': y.tolist()
                        }
                    }
            }
            self.result = self.result + [iou_mean]
        else:
            dict_ret = {}

        return {}

    def get_result(self) -> dict:
        r = np.array(self.result)
        r_float = float(r.mean())
        self.cnt = self.cnt + 0.5
        ratio = np.tan(r_float)
        y = self.x * ratio
        dict_ret = {
            'overall': {
                    'customized_iou': r_float,
                    'float_file_metric': self.cnt,
                    'curve_file_metric': {
                        'x': self.x.tolist(),
                        'y': y.tolist()
                    }
                }
        }
        return dict_ret

