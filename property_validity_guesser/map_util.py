MAP_BBOX_OVERSIZE_THRESHOLD = 0.05 ## 0.05 in decimal degree is around +/- 5.5km in equator

class MapUtil(object):
    oversize_threshold = MAP_BBOX_OVERSIZE_THRESHOLD

    @staticmethod
    def bbox_oversize_predictor(NE_bounding_box, SW_bounding_box):
        '''
        :param NE_bounding_box: north east coordinate, tuple in format (latitude, longitude)
        :param SW_bounding_box: south west cooridnate, tuple in format (latitude, longitude)
        :return: boolean, if the bounding box is oversize or not
        :return: boolean, if the bounding box is oversize or not
        '''

        latitude1, longitude1 = NE_bounding_box
        latitude2, longitude2 = SW_bounding_box

        if abs(latitude1 - latitude2) > MapUtil.oversize_threshold and \
           abs(longitude1 - longitude2) > MapUtil.oversize_threshold:
            return True
        else:
            return False

