import ReadWrite_h5
import numpy as np
import pandas as pd
import os
import glob
import glob

def math_month_mean(path, outputpath):
    tifpaths = glob.glob(os.path.join(path, '*.tif'))

    for i in range(1, 13):
        data_list = []
        tifName = None
        PROJECTION = None
        TRANSFORM = None
        for tifpath in tifpaths:

            if '2020' + str(i).rjust(2, '0') in tifpath:
                tifDataset = ReadWrite_h5.get_tifDataset(tifpath)
                PROJECTION, TRANSFORM = ReadWrite_h5.get_GeoInformation(tifDataset)
                imgx, imgy = ReadWrite_h5.get_RasterXY(tifDataset)
                tifArr = ReadWrite_h5.get_RasterArr(tifDataset, imgx, imgy)
                tifArr = tifArr[:4, :, :]
                data_list.append(tifArr)
        tifName = '2020-' + str(i).rjust(2, '0')

        data_list = np.array(data_list)
        data_list[data_list == -1] = np.nan
        print(data_list.shape)

        result = np.nanmean(data_list, axis=0)
        print(result.shape)
        if result.shape == ():

            continue

        ReadWrite_h5.write_tif(result, PROJECTION, TRANSFORM, os.path.join(outputpath, tifName + '.tif'))
        del result

def math_year_mean(path, outputpath):
    path = r'J:\23_06_lunwen\China-linear\hecheng'

    PROJECTION = None
    TRANSFORM = None
    # a = np.array([[[[1,1,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]]])
    # b = np.array([[[[1,np.nan,3],[1,2,5],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]]])
    # c = []
    # c.append(a)
    # c.append(b)
    # print(np.nanmean(np.array(c), axis=0))
    # exit(0)

    """
    for i in range(1, 13):
        data_list = []
        for temp in os.listdir(path):
    
            if os.path.splitext(temp)[1] != '.tif':
                continue
    
            if '2021' + str(i).rjust(2, '0') not in temp:
                continue
    
            print(temp)
            dataset = ReadWrite_h5.get_tifDataset(os.path.join(path, temp))
    
            pro, trans = ReadWrite_h5.get_GeoInformation(dataset)
            PROJECTION = pro
            TRANSFORM = trans
            imgx, imgy = ReadWrite_h5.get_RasterXY(dataset)
            arr = ReadWrite_h5.get_RasterArr(dataset, imgx, imgy)
    
            data_list.append(arr)
    
        math_month_mean(np.array(data_list), PROJECTION, TRANSFORM, r'J:\澜湄2016-2021\year_mean\Lanmei\2021-' + str(i).rjust(2, '0') + '_Lanmei.tif')
    """

    # 计算年平均
    data_list = []
    for temp in os.listdir(path):

        if os.path.splitext(temp)[1] != '.tif':
            continue

        print(os.path.basename(temp))
        dataset = ReadWrite_h5.get_tifDataset(os.path.join(path, temp))

        pro, trans = ReadWrite_h5.get_GeoInformation(dataset)
        PROJECTION = pro
        TRANSFORM = trans
        imgx, imgy = ReadWrite_h5.get_RasterXY(dataset)
        arr = ReadWrite_h5.get_RasterArr(dataset, imgx, imgy)

        data_list.append(arr)

    math_month_mean(np.array(data_list), PROJECTION, TRANSFORM, r'J:\23_06_lunwen\China-linear\year_mean\xia_mean.tif')


if __name__ == '__main__':

    math_month_mean(r'J:\AODinversion\AOD的TOA数据\sentinel-2_mosic_kny', r'J:\AODinversion\AOD的TOA数据\sentinel-2_0.001\kny')
