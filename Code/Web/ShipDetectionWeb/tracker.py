from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        # Khởi tạo ID tiếp theo là 0
        self.nextObjectID = 0
        # Dictionary lưu trữ ID và tọa độ tâm (Centroid)
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # Dictionary lưu lịch sử đường đi (Trail) cho hiệu ứng WOW
        # Key: ID, Value: List các điểm [(x,y), (x,y)...]
        self.trails = OrderedDict()

        # Số frame tối đa vật thể được phép mất tích trước khi xóa ID
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # Đăng ký vật thể mới
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trails[self.nextObjectID] = [tuple(centroid)]  # Khởi tạo trail
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Xóa vật thể
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trails[objectID]

    def update(self, rects):
        # Kiểm tra nếu không có box nào được detect
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.trails

        # Tính toán tâm (centroid) cho các box mới detect
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Nếu chưa có vật thể nào đang theo dõi, đăng ký hết
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # Nếu đã có vật thể, thực hiện thuật toán so khớp khoảng cách
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Tính khoảng cách giữa các tâm cũ và tâm mới
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # [WOW EFFECT] Cập nhật đường đi (Trail)
                # Chỉ giữ lại 30 điểm gần nhất để đuôi không quá dài
                self.trails[objectID].append(tuple(inputCentroids[col]))
                if len(self.trails[objectID]) > 30:
                    self.trails[objectID].pop(0)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Xử lý vật thể biến mất
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Đăng ký vật thể mới xuất hiện
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects, self.trails