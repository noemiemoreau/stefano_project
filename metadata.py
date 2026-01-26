import os
import numpy as np
import pandas as pd
from   pathlib import Path
import re
from   utils import list_subdir_filter as lsd, flatten

class metadata:

# updated for cookiecutter use... good luck to us

    
    def __init__(self):
        if os.path.exists('/projects/ag-bozek/sugliano/dlbcl/'):
            self.cookiecutter_dir = '/projects/ag-bozek/sugliano/dlbcl/'
        elif os.path.exists('/projects/ag-bozek/dlbcl/'):
            self.cookiecutter_dir = '/projects/ag-bozek/dlbcl/'            



        self.base_dir = os.path.join(self.cookiecutter_dir, 'data/interim')
        self.subfolders = {
            'images'      : 'Run0002',
            'previews'    : 'previews',
            'dapi_masks'  : 'previews/dapi_mask_visualization',
            'segmented'   : 'segmented',
            'regionprops' : 'regionprops',
            'aligned'     : 'aligned',
            'bbox_coords' : 'bbox_coords',
            'distance_matrix': 'distance_matrix',
            'edge_lists'  : 'edge_lists',
            'label_shuffle': 'label_shuffle',
            'roi'         : 'roi',
            'results'     : 'results'
        }


        self.folders = {k:os.path.join(self.base_dir, v) for k,v in self.subfolders.items()}


        self.folders['data_ext'] = os.path.join(self.cookiecutter_dir, 'data/external')
        self.folders['images'] = os.path.join(self.cookiecutter_dir, 'data/raw/Run0002')

        self.markers = [
            'CD8A',     'CD31',  'LAMIN_B',
            'CD11B',    'CD3D',  'CD20',
            'CD163',    'CD68',  'CD204',
            'CD4',      'FOXP3', 'CD138',
            'LAMIN_AC', 'PDL_1', 'CD56'
        ]

        self.markers_with_dapi = [
            ['DAPI1', 'CD8A',     'CD31',  'LAMIN_B'],
            ['DAPI2', 'CD11B',    'CD3D',  'CD20'],
            ['DAPI3', 'CD163',    'CD68',  'CD204'],
            ['DAPI4', 'CD4',      'FOXP3', 'CD138'],
            ['DAPI5', 'LAMIN_AC', 'PDL_1', 'CD56']
        ]

        self.markers_position = {}
        for i in range(5):
            for j in range(4):
                self.markers_position[self.markers_with_dapi[i][j]] = {'coords':[i,j], 'index':i*4+j}

        self.markers_selected = {
            'indices':np.bool([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])
        }
        self.markers_selected['names'] = np.array(flatten(self.markers_with_dapi))[self.markers_selected['indices']]
        
        self.marker_biology = {
            'CD8A'    : 'T cells',
            'CD31'    : 'endothelial cells/vessels',
            'LAMIN_B' : 'nuclear membrane',
            'CD11B'   : 'dendritic cells',
            'CD3D'    : 'T cells',
            'CD20'    : 'B cells',
            'CD163'   : 'macrophages',
            'CD68'    : 'macrophages',
            'CD204'   : 'macrophages',
            'CD4'     : 'T cells',
            'FOXP3'   : 'T reg cells',
            'CD138'   : 'plasma cells, NK cells',
            'LAMIN_AC': 'nuclear membrane',
            'PDL1'    : 'PD-L1',
            'CD56'    : 'NK cells'
        }


        self.color_mappings = {
            'full_classification':{

                'Tumor': '#DD1100',
                'B cell NOS': '#FF9A9E',
                'Endothelial Cell': '#002F77',
                'Monocyte NOS': '#207960',
                'PDL1': '#840000',
                'M0/M1 Macrophage':'#B586D8',
                'M2 Macrophage':'#753678',

                'CD4+ T-helper cell': '#2C72B8',
                'CD4+ activated T-helper cell': '#85C2FF',
                'Double positive T-cell': '#4FFFFF',
                'CD8+ cytotoxic T-cell': '#2FBD45',
                'CD8+ activated cytotoxic T-cell': '#38FF8E',

                'T-cell NOS': '#A86A27',
                'T-cell NOS activated': '#F5F500',
                'T-regulatory cell (T-reg)': '#FFAA00',

                'other': '#D6D6D6',
                'unusual': '#7D7D7D',
                'Negative': '#282828'
            },
            'simple_classification':{
                'Tumor':'#DD1100',
                'B-Cell':'#FF9A9E',
                'Endothelial Cell': '#002F77',
                'M0/M1 Macrophage':'#B586D8',
                'M2 Macrophage':'#753678',
                'CD4+':'#2C72B8',
                'CD4+8+':'#32E5ED',
                'CD8+':'#2FBD45',
                'T-Cell':'#A86A27',
                'other': '#D6D6D6',
                'unusual': '#7D7D7D',
                'Negative': '#282828'
            }
        }

        # self.marker_placement = {
        #     'CD8A'    : {'what': 'T cells'},
        #     'CD31'    : {'what': 'endothelial cells/vessels'},
        #     'LAMIN_B' : {'what': 'nuclear membrane'},
        #     'CD11B'   : {'what': 'dendritic cells'},
        #     'CD3D'    : {'what': 'T cells'},
        #     'CD20'    : {'what': 'B cells'},
        #     'CD163'   : {'what': 'macrophages'},
        #     'CD68'    : {'what': 'macrophages'},
        #     'CD204'   : {'what': 'macrophages'},
        #     'CD4'     : {'what': 'T cells'},
        #     'FOXP3'   : {'what': 'T reg cells'},
        #     'CD138'   : {'what': 'plasma cells}, NK cells'},
        #     'LAMIN_AC': {'what': 'nuclear membrane'},
        #     'PDL1'    : {'what': 'PD-L1'},
        #     'CD56'    : {'what': 'NK cells'}
        # }

        self.classification_grouping_dict = {

            'B cell NOS':'B-Cell',
            'CD4+ T-helper cell':'CD4+',
            'CD4+ activated T-helper cell':'CD4+',
            'CD8+ activated cytotoxic T-cell':'CD8+',
            'CD8+ cytotoxic T-cell':'CD8+',
            'Double positive T-cell':'CD4+8+',
            'Endothelial cell':'Endothelial Cell',
            'M0/M1 macrophage':'M0/M1 Macrophage',
            'M2 macrophage':'M2 Macrophage',
            'Monocyte NOS':'other',
            'PDL1':'other',
            'T-cell NOS':'T-Cell',
            'T-cell NOS activated':'T-Cell',
            'T-regulatory cell (T-reg)':'T-Cell',
            'Tumor':'Tumor',
            'Negative':'Negative',
            'unusual':'unusual',
            'everything else':'other'

        }
        self.chosen_ids = [
            'DLBCL1_0', 'DLBCL10_0', 'DLBCL100_1', 'DLBCL101_2', 'DLBCL102_1',
            'DLBCL104_0', 'DLBCL106_1', 'DLBCL11_0', 'DLBCL12_2', 'DLBCL13_2',
            'DLBCL14_2', 'DLBCL15_2', 'DLBCL16_1', 'DLBCL17_0', 'DLBCL18_1',
            'DLBCL19_2', 'DLBCL2_1', 'DLBCL20_1', 'DLBCL21_0', 'DLBCL22_2',
            'DLBCL23_0', 'DLBCL24_2', 'DLBCL25_0', 'DLBCL26_0', 'DLBCL27_2',
            'DLBCL28_0', 'DLBCL29_1', 'DLBCL3_2', 'DLBCL30_0', 'DLBCL31_0',
            'DLBCL32_2', 'DLBCL33_1', 'DLBCL34_2', 'DLBCL35_2', 'DLBCL36_2',
            'DLBCL37_2', 'DLBCL38_1', 'DLBCL39_1', 'DLBCL4_1', 'DLBCL40_1',
            'DLBCL41_2', 'DLBCL42_1', 'DLBCL43_1', 'DLBCL44_1', 'DLBCL45_2',
            'DLBCL46_1', 'DLBCL5_1', 'DLBCL59_2', 'DLBCL6_2', 'DLBCL60_0',
            'DLBCL61_0', 'DLBCL62_2', 'DLBCL63_1', 'DLBCL64_0', 'DLBCL65_1',
            'DLBCL66_2', 'DLBCL67_1', 'DLBCL68_0', 'DLBCL69_0', 'DLBCL7_1',
            'DLBCL70_2', 'DLBCL71_1', 'DLBCL72_2', 'DLBCL73_0', 'DLBCL74_0',
            'DLBCL75_2', 'DLBCL76_2', 'DLBCL77_0', 'DLBCL78_2', 'DLBCL79_1',
            'DLBCL8_0', 'DLBCL80_1', 'DLBCL81_1', 'DLBCL82_0', 'DLBCL83_1',
            'DLBCL84_1', 'DLBCL85_0', 'DLBCL86_2', 'DLBCL87_0', 'DLBCL88_2',
            'DLBCL89_2', 'DLBCL9_2', 'DLBCL90_1', 'DLBCL91_2', 'DLBCL92_1',
            'DLBCL93_1', 'DLBCL94_2', 'DLBCL95_2', 'DLBCL96_2', 'DLBCL97_2',
            'DLBCL98_1', 'DLBCL99_2', 'MCL1_0', 'MCL10_2', 'MCL11_2',
            'MCL12_2', 'MCL13_2', 'MCL14_2', 'MCL15_2', 'MCL17_0', 'MCL18_2',
            'MCL2_1', 'MCL20_0', 'MCL21_2', 'MCL22_2', 'MCL23_0', 'MCL24_0',
            'MCL25_2', 'MCL26_1', 'MCL27_0', 'MCL28_0', 'MCL29_1', 'MCL3_2',
            'MCL30_0', 'MCL31_0', 'MCL32_2', 'MCL33_2', 'MCL34_2', 'MCL35_1',
            'MCL36_2', 'MCL37_2', 'MCL38_2', 'MCL39_1', 'MCL4_1', 'MCL40_1',
            'MCL41_1', 'MCL42_2', 'MCL43_2', 'MCL44_0', 'MCL45_1', 'MCL46_2',
            'MCL47_0', 'MCL48_0', 'MCL49_0', 'MCL5_1', 'MCL50_1', 'MCL51_0',
            'MCL52_0', 'MCL53_0', 'MCL54_2', 'MCL6_2', 'MCL7_0', 'MCL8_1',
            'MCL9_1']


        self.dates = dict(zip(
            [f'cycle{i}' for i in range(5)],
            [os.path.basename(f) for f in lsd(self.folders['images'])]
        ))


        for f in self.folders.values():
            Path(f).mkdir(exist_ok=True, parents=True)

        self.id_matching = pd.read_csv(
            '/projects/ag-bozek/sugliano/dlbcl/data/external/Run0002_Core_sampleID_matching.csv',
            dtype='object'
        ).dropna(how='all')

        self.id_matching.columns = map(str.lower, self.id_matching.columns)

        self.id_matching['unique_id'] = self.id_matching.core_id + '_' + \
            self.id_matching.groupby('core_id').cumcount().astype('str')

        self.id_matching.loc[self.id_matching.pg_cluster == 'OH4', 'pg_cluster'] = 'PG4'


        self.id_matching['pg_cluster'] = [
            re.sub('PGG', 'PG', str(c).upper())
            for c in self.id_matching.pg_cluster
        ]

        clinical_metadata = pd.read_csv(
            '/projects/ag-bozek/sugliano/dlbcl/data/external/CLINICAL_DATA_MDG_20250403.csv',
            dtype='object'
        ).drop_duplicates()

        clinical_metadata.columns = map(str.lower, clinical_metadata.columns)


        self.id_matching = pd.merge(
            self.id_matching,
            clinical_metadata,
            left_on='core_id',
            right_on='id',
            how='left'
        ).drop('id', axis=1)
