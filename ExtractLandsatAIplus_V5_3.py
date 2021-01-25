###############################################################################################################
#   Revision V5.3:
#   - Fixing the bug in picking multiple metrics per GLCM feature
###############################################################################################################
#   9/15/2020, Shahriar S. Heydari

import ee
import time
import numpy as np

ee.Initialize()

assets = ee.data.getList({'id': 'users/shshheydari/Final84Blocks'})
GCbucketName = 'shahriarsh_temp'  # Google Cloud bucket to save output result
extract_periods = [('2005-01-01','2006-01-01'),('2006-01-01','2007-01-01'),('2007-01-01','2008-01-01'),
                   ('2008-01-01','2009-01-01'),('2009-01-01','2010-01-01'),('2010-01-01','2011-01-01'),
                   ('2011-01-01','2012-01-01'),('2012-01-01','2013-01-01'),('2013-01-01','2014-01-01'),
                   ('2014-01-01','2015-01-01'),('2015-01-01','2016-01-01'),('2016-01-01','2017-01-01'),
                   ('2017-01-01','2018-01-01'),('2018-01-01','2019-01-01'),('2019-01-01','2020-01-01')]
extract_periods = [('2010-05-01','2010-06-01')]
selected_assets = list(assets[i] for i in [0])
addSpectralIndices = True  # if true, below SPIs are added to the output data.
maskCloud = True
SPIs2Create = ['BSI','DD','ENDISI','NDVI']#['BSI','DD','ENDISI','MNDWI','MSAVI2','NDVI','NLI','VSDI']
addGLCM = True  # if true, below GLCM_features are added to the output data.
# NOTE: only one base, neoghborhood, and qlevel is allowed per entry but metrics can be multiple.
GLCM_features = [
    {'band':'blue', 'neighborhood':5, 'qlevel':64, 'metrics':['_diss','_ent','_savg','_var']},
    {'band':'DD',  'neighborhood':5, 'qlevel':64, 'metrics':['_diss','_ent','_savg','_var']},
    {'band':'ENDISI',  'neighborhood':5, 'qlevel':64, 'metrics':['_diss','_ent','_savg','_var']},
    {'band':'NIR',  'neighborhood':5, 'qlevel':64, 'metrics':['_diss','_ent','_savg','_var']},
    {'band': 'blue', 'neighborhood': 15, 'qlevel': 64, 'metrics': ['_diss', '_ent', '_savg', '_var']},
    {'band': 'DD', 'neighborhood': 15, 'qlevel': 64, 'metrics': ['_diss', '_ent', '_savg', '_var']},
    {'band': 'ENDISI', 'neighborhood': 15, 'qlevel': 64, 'metrics': ['_diss', '_ent', '_savg', '_var']},
    {'band': 'NIR', 'neighborhood': 15, 'qlevel': 64, 'metrics': ['_diss', '_ent', '_savg', '_var']}
                 ]

addTerrainBands = True                  # if true, will add terrain data (from SRTM) and rain data (from PRISM)
addEcoRegion = False                    # if true, ecoregion is added as a band
addPRISMdata = False                    # if true, will add image year's rain and temperature statistics
addPRISMhistorical = False               # if true, a separate PRISM long-term dataset is generated
enlargedBoxes = False                   # if true, read the box boundaries from enlarged bounding box directory
genTestData = False                     # if true, if doesn't run maskTimePeriod
patch_size = 37
# if genTestData:
#     if enlargedBoxes:
#         assets = ee.data.getList({'id': 'users/shshheydari/LCmapTestAreas'})
#         patch_size = 50
#     else:
#         assets = ee.data.getList({'id': 'users/shshheydari/CorrectedLCmaps-3band'})
#         patch_size = 37
# else:
#     assets = ee.data.getList({'id': 'users/shshheydari/CorrectedLCmaps-3band'})
#     patch_size = 37

# Will combine scenes from Landsat 5, 7, and 8. But the Landsat 8 bands are different. So I defined below lists
# to match corresponding bands in ETM/OLI and make unified names for bands in output feature list
ETM_selected_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa', 'radsat_qa']
OLI_selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa', 'radsat_qa']
processed_bands = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2', 'pixel_qa', 'radsat_qa']
spectral_bands = ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2']
Landsat5SR = ee.ImageCollection("LANDSAT/LT05/C01/T1_SR")
Landsat7SR = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")
Landsat8SR = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
SRTM = ee.Image("USGS/SRTMGL1_003")
monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
PRISM = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")
# Historical data that we want to use for climate variables
rain_h = PRISM.filterDate('1990-01-01','2019-12-31').select('ppt')
temp_h = PRISM.filterDate('1990-01-01','2019-12-31').select('tmean')
# RESOLVE is a global ecoregion identifier dataset
RESOLVE_ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
box = []
proj = []
scale = []
inputMap = []
null_image = ee.Image(0)
#null_GLCM_image = ee.Image(0)

# Below bands are generated in the output array
bands = ['DOY', 'sensor']
if addEcoRegion:
    bands = bands + ['ECO_ID']
if addTerrainBands:
    bands = bands + ['elevation', 'slope', 'aspect']
if addPRISMdata:
    bands = bands + ['temp_'+x for x in monthNames] + ['rain_'+x for x in monthNames]
bands = bands + ['blue', 'green', 'red', 'NIR', 'SWIR1', 'SWIR2']
if addSpectralIndices:
    bands = bands + SPIs2Create
if not maskCloud:
    bands = bands + ['pixel_qa', 'radsat_qa']
if addGLCM:
    for GLCM_feature in GLCM_features:
        bands = bands + [GLCM_feature['band'] + x + '_'+str(GLCM_feature['neighborhood'])+'x'+str(GLCM_feature['qlevel']) for x in GLCM_feature['metrics']]

# This function masks those pixels that fall before 'FirstYear' of after 'LastYear'
def maskTimePeriod(image):
    year = image.date().get('year')
    yearBand = ee.Image.constant(year).uint16().rename('imageYear').reproject(proj).clip(box)
    mask = (yearBand.gte(inputMap.select('firstYear'))).And(yearBand.lte(inputMap.select('lastYear')))
    return image.updateMask(mask)

# This function masks cloud/cloud shadow and saturated pixels
def maskLandsatSR(image):
    # Bits 3, 5, 7, and 9  in Landsat7/8 show non-clear conditions
    cloudShadowBitMask = 2 ** 3
    cloudsBitMask = 2 ** 5
    cloudsConfBitMask = 2 ** 7
    cirrusConfBitMask = 2 ** 9
    # Get the pixel QA bands
    qa = image.select('pixel_qa')
    radsat = image.select('radsat_qa')
    # All flags should be set to zero, indicating clear conditions.
    mask = (qa.bitwiseAnd(cloudShadowBitMask).eq(0)).And(qa.bitwiseAnd(cloudsBitMask).eq(0)) \
        .And(qa.bitwiseAnd(cloudsConfBitMask).eq(0)).And(qa.bitwiseAnd(cirrusConfBitMask).eq(0)) \
        .And(radsat.eq(0))
    image = image.updateMask(mask)

    return image

# This function adds some auxiliary information to each pixel (DOY, topographic data, etc.)
def myAddBands(image):
    p1 = image.get('system:index')
    p2 = image.get('system:time_start')
    image_date = image.date()
    # Find spectral band values less than zero or greater than 10000 and zero them
    spectralImage = image.select(spectral_bands).divide(10000)
    mask2 = spectralImage.lt(1).And(spectralImage.gt(0))
    spectralImage = spectralImage.multiply(mask2)
    image = spectralImage.addBands(image.select(['sensor', 'pixel_qa', 'radsat_qa']))
    # Add other bands
    doy = image_date.getRelative('day', 'year')
    doyBand = ee.Image.constant(doy).uint16().rename('DOY')
    image = image.addBands(doyBand)
    ecoBand = RESOLVE_ecoregions.reduceToImage(properties=['ECO_ID'], reducer=ee.Reducer.first()).rename('ECO_ID')
    if addEcoRegion:
        image = image.addBands(ecoBand)
    if addTerrainBands:
        terrainBands = ee.Terrain.products(SRTM)
        image = image.addBands(terrainBands)

    if addPRISMdata:
        image_year = image_date.getRange('year')
        rain_stats = PRISM.filterDate(image_year).select('ppt').toBands()
        temp_stats = PRISM.filterDate(image_year).select('tmean').toBands()
        rain_stats = rain_stats.select(rain_stats.bandNames(), ['rain_' + x for x in monthNames])
        temp_stats = temp_stats.select(temp_stats.bandNames(), ['temp_' + x for x in monthNames])
        image = image.addBands(rain_stats).addBands(temp_stats)

    return image.reproject(proj).clip(box).setMulti({
        'system:index': p1,
        'system:time_start': p2
    })

# This functions calculate spectral indices and add it to the image array
def addSPI(image):
    if 'NDVI' in SPIs2Create:
        image = image.addBands(image.expression('(b("NIR")-b("red"))/(b("NIR")+b("red"))').rename('NDVI'))
    if 'MSAVI2' in SPIs2Create:
        image = image.addBands(image.expression('0.5*(1+2*b("NIR")-((2*b("NIR")+1)**2-8*(b("NIR")-b("red")))**0.5)')
                  .rename('MSAVI2'))
    if ('MNDWI' in SPIs2Create) or ('ENDISI' in SPIs2Create):
        image = image.addBands(image.expression('(b("green")-b("SWIR1"))/(b("green")+b("SWIR1"))').rename('MNDWI'))
    if 'NLI' in SPIs2Create:
        image = image.addBands(image.expression('(b("NIR")**2-b("red"))/(b("NIR")**2+b("red"))').rename('NLI'))
    if 'BSI' in SPIs2Create:
        image = image.addBands(image.expression('(b("SWIR1")+b("red")-b("NIR")-b("blue"))/(b("SWIR1")+b("red")+b("NIR")+b("blue"))')
                  .rename('BSI'))
    if 'DD' in SPIs2Create:
        image = image.addBands(image.expression('(b("NIR")**2+b("red")**2)**0.5').rename('DD'))
    if 'VSDI' in SPIs2Create:
        image = image.addBands(image.expression('1-(b("SWIR1")+b("red")-2*b("blue"))').rename('VSDI'))
    if 'ENDISI' in SPIs2Create:
        BLUE_mean = image.select('blue').reduceRegion(reducer=ee.Reducer.mean(),
                                                      geometry=box, scale=scale,maxPixels=1e9).toImage()
        mndwi = image.select('MNDWI')
        mean2 = mndwi.multiply(mndwi).reduceRegion(reducer=ee.Reducer.mean(),
                                                   geometry=box, scale=scale,maxPixels=1e9).toImage()
        mean3 = image.select('SWIR1').divide(image.select('SWIR2')).reduceRegion(reducer=ee.Reducer.mean(),
                                                                                 geometry=box, scale=scale,
                                                                                 maxPixels=1e9).toImage()
        endisi_alpha = BLUE_mean.multiply(2).divide(mean2.add(mean3)).rename('alpha')
        image = image.addBands(endisi_alpha)
        image = image.addBands(image.expression(
            '(b("blue") - b("alpha")*((b("SWIR1")/b("SWIR2"))+b("MNDWI")**2)) / (b("blue") + b("alpha")*((b("SWIR1")/b("SWIR2"))+b("MNDWI")**2))')
            .rename('ENDISI'))
    return image

# This functions calculate GLCM features and add it to the image array
def add_GLCM(image):
    def genGLCM(feature):
        sel_band = image.select(feature['band'])
        GLCM_f = [feature['band']+x for x in feature['metrics']]
        sel_GLCM_features = [feature['band']+x+'_'+str(feature['neighborhood'])+'x'+str(feature['qlevel']) for x in feature['metrics']]
        min = ee.Number(sel_band.reduceRegion(reducer=ee.Reducer.min(),
                                              geometry=box, scale=scale, maxPixels=10e9).get(feature['band']))
        max = ee.Number(sel_band.reduceRegion(reducer=ee.Reducer.max(),
                                              geometry=box, scale=scale, maxPixels=10e9).get(feature['band']))
        return ee.Algorithms.If(max,
                                ee.Algorithms.If(max.gt(min),
                                                 sel_band.unitScale(min, max).multiply(feature['qlevel']).toInt()
                                                 .glcmTexture(size=feature['neighborhood']).select(GLCM_f).rename(sel_GLCM_features),
                                                 null_GLCM_image.reproject(proj).clip(box).rename(sel_GLCM_features)),
                                null_GLCM_image.reproject(proj).clip(box).rename(sel_GLCM_features))

    for feature in GLCM_features:
        null_GLCM_image = null_image
        for i in range(len(feature['metrics']) - 1):
            null_GLCM_image = null_GLCM_image.addBands(null_image)
        image = image.addBands(genGLCM(feature))
    return image

########################################################################################################
# Main loop

# Create daily datasets
for asset in selected_assets:

    # A) create array image time series data for each setting of GLCM parameters and each time period
    for period in extract_periods:
        start = period[0]
        end = period[1]
        id_str = asset['id']
        print('Processing asset {} for period {} ...'.format(id_str, period))
        if genTestData and enlargedBoxes:
            index = id_str.lower().find('bbox') + 4
        else:
            index = id_str.lower().find('samp') + 4
        samp_id = id_str[index:index + 7]
        ecoregion = int(samp_id[0:2])
        # Read the input map
        inputMap = ee.Image(asset['id'])
        box = inputMap.geometry()
        proj = inputMap.projection()
        scale = proj.nominalScale().getInfo()

        # Mask invalid landcovers (less than 21 or greater than 27)
        mask = inputMap.select(['b1']).lt(28).multiply(inputMap.select(['b1']).gt(20))
        inputMap = inputMap.select(['b2', 'b3']).addBands(inputMap.select('b1').multiply(mask))
        inputMap = inputMap.select(['b1', 'b2', 'b3'], ['landcover', 'firstYear', 'lastYear'])

        # Build merged Landsat collection for the specified time period and region, and unify the band names
        LandsatCol1 = Landsat5SR.filterDate(start, end).filterBounds(box).select(ETM_selected_bands,processed_bands)
        LandsatCol2 = Landsat7SR.filterDate(start, end).filterBounds(box).select(ETM_selected_bands,processed_bands)
        LandsatCol3 = Landsat8SR.filterDate(start, end).filterBounds(box).select(OLI_selected_bands,processed_bands)

        # Add sensor type (Landsat 5/7/8)
        LandsatCol1 = LandsatCol1.map(lambda image: image.addBands(ee.Image.constant(5).rename('sensor')))
        LandsatCol2 = LandsatCol2.map(lambda image: image.addBands(ee.Image.constant(7).rename('sensor')))
        LandsatCol3 = LandsatCol3.map(lambda image: image.addBands(ee.Image.constant(8).rename('sensor')))

        # Process images using above defined functions
        LandsatCol = LandsatCol1.merge(LandsatCol2).merge(LandsatCol3).sort('system:time_start')
        if not genTestData:
            LandsatCol = LandsatCol.map(maskTimePeriod)
        if maskCloud:
            LandsatCol = LandsatCol.map(maskLandsatSR)
        LandsatCol = LandsatCol.map(myAddBands)
        if SPIs2Create:
            LandsatCol = LandsatCol.map(addSPI)
        if addGLCM:
            LandsatCol = LandsatCol.map(add_GLCM)
        LandsatCol = LandsatCol.select(bands)
        NBands = LandsatCol.first().bandNames().length().getInfo()

        # Clip every scene to the selected region, reproject it, and aggregate all clipped images to a single image
        imageT = ee.Image(0).reproject(proj).clip(box)
        def accumulate1(image, imageT):
            return (
                ee.Image(imageT).addBands(image).unmask())#.reproject(proj)).unmask())  # .int().reproject(proj)).unmask())
        imageOfSeries = ee.Image(LandsatCol.iterate(accumulate1, imageT)).slice(1).addBands(inputMap.select('landcover'))

        # Aggregate bands and scenes information (Landsat IDs) to a feature collection (to be used later when
        # reading image stack)
        timeT = ee.List([ee.Feature(None, {'imageTag': bands})])
        def accumulate2(image, timeT):
            return (ee.List(timeT).add(ee.Feature(None, {'imageTag': image.get('system:index')})))
        timeTags = ee.List(LandsatCol.iterate(accumulate2, timeT))

        # NFeatures = timeTags.length().getInfo() - 1
        # Generate output file name
        if genTestData:
            AssetName = 'samp' + samp_id + '_' + 'test_' + start + '_' + end + '_' + str(NBands) + 'band'
        else:
            AssetName = 'samp' + samp_id + '_' + start + '_' + end + '_' + str(NBands) + 'band'
        AssetName = AssetName + '_AI'

        # Write scene IDs to a CSV file
        task = ee.batch.Export.table.toCloudStorage(
            collection=ee.FeatureCollection(timeTags),
            description=AssetName + 'timeTags_',
            bucket=GCbucketName,
            fileNamePrefix=AssetName + 'timeTags_',
            selectors=['imageTag']
        )
        task.start()

        #tDepth = NBands * NFeatures + 1
        # Write stack of images to a TFrecord file set
        task = ee.batch.Export.image.toCloudStorage(
            image=imageOfSeries.float(),
            description=AssetName,
            fileNamePrefix=AssetName,
            bucket=GCbucketName,
            scale=scale,
            # dimensions = '333x333',
            region=box.getInfo()['coordinates'],
            fileFormat='TFRecord',
            formatOptions={
                'patchDimensions': [patch_size, patch_size],
                # 'tensorDepths': [tDepth],
                'collapseBands': True,
                # The output combined band will have the name of the first band, i.e. DOY
                'compressed': True
            }
        )
        task.start()
        print('Waiting for task completion...')
        while (task.status()['state'] not in ['COMPLETED', 'FAILED']):
            time.sleep(60)
        print('Task {}.'.format(task.status()['state']))

    # B) Create PRISM historical dataset
    if addPRISMhistorical:
        id_str = asset['id']
        print('Generating asset {} historical climate data ...'.format(id_str))
        if genTestData and enlargedBoxes:
            index = id_str.lower().find('bbox') + 4
        else:
            index = id_str.lower().find('samp') + 4
        samp_id = id_str[index:index + 7]

        # Read the input map
        inputMap = ee.Image(asset['id'])
        box = inputMap.geometry()
        proj = inputMap.projection()
        scale = proj.nominalScale().getInfo()

        def extract(im_h, set):
            if set == 'temp':
                NmonthBands = [['Ntemp_'+x for x in monthNames]]
                NyearBands = [['Ntemp_'+x for x in ['_mean', '_min', '_max']]]
            else:  # set == 'rain'
                NmonthBands = [['Nrain_'+x for x in monthNames]]
                NyearBands = [['Nrain_'+x for x in ['_mean', '_min', '_max']]]

            im1 = im_h.toArray().clip(box)
            N_rows = im1.arrayLength(0)
            imageAxis = 0

            # Generate monthly aggregates (over rows)
            im2 = im1.arraySlice(imageAxis, 0, N_rows, 12)  # extract January data
            # add other months in columns
            for i in range(1, 12):
                im2 = im2.arrayCat(im1.arraySlice(imageAxis, i, im1.arrayLength(0), 12), 1)
            Monthly_historical = im2.arrayReduce(ee.Reducer.mean(), [0]).arrayProject([1]).arrayFlatten(NmonthBands)

            # Generate yearly aggregates (over columns)
            if set == 'temp':
                im3 = im2.arrayReduce(ee.Reducer.mean(), [1])
            else: # set == 'rain'
                im3 = im2.arrayReduce(ee.Reducer.sum(), [1])
            min = im3.arrayReduce(ee.Reducer.min(), [0])
            max = im3.arrayReduce(ee.Reducer.max(), [0])
            mean = im3.arrayReduce(ee.Reducer.mean(), [0])
            Yearly = mean.arrayCat(min.arrayCat(max, 1), 1).arrayProject([1]).arrayFlatten(NyearBands)
            return Monthly_historical.addBands(Yearly)


        historical_stats = extract(temp_h, 'temp').addBands(extract(rain_h, 'rain')).reproject(proj).clip(box)
        Bands = historical_stats.bandNames()
        # ['Ntemp_Jan', 'Ntemp_Feb', 'Ntemp_Mar', 'Ntemp_Apr', 'Ntemp_May', 'Ntemp_Jun', 'Ntemp_Jul', 'Ntemp_Aug',
        #  'Ntemp_Sep', 'Ntemp_Oct', 'Ntemp_Nov', 'Ntemp_Dec', 'Ntemp_mean', 'Ntemp_min', 'Ntemp_max',
        #  'Nrain_Jan', 'Nrain_Feb', 'Nrain_Mar', 'Nrain_Apr', 'Nrain_May', 'Nrain_Jun', 'Nrain_Jul', 'Nrain_Aug',
        #  'Nrain_Sep', 'Nrain_Oct', 'Nrain_Nov', 'Nrain_Dec', 'Nrain_mean', 'Nrain_min', 'Nrain_max']
        NBands = Bands.length().getInfo()
        AssetName = 'samp' + samp_id + '_historicalClimateData_' + str(NBands) + 'band'

        task = ee.batch.Export.image.toCloudStorage(
            image=historical_stats.float(),
            description=AssetName,
            fileNamePrefix=AssetName,
            bucket=GCbucketName,
            scale=scale,
            # dimensions = '333x333',
            region=box.getInfo()['coordinates'],
            fileFormat='TFRecord',
            formatOptions={
                'patchDimensions': [patch_size, patch_size],
                # 'tensorDepths': [tDepth],
                'collapseBands': True,
                # The output combined band will have the name of the first band, i.e. DOY
                'compressed': True
            }
        )
        task.start()
        print('Waiting for task completion...')
        while (task.status()['state'] not in ['COMPLETED', 'FAILED']):
            time.sleep(60)
        print('Task {}.'.format(task.status()['state']))
