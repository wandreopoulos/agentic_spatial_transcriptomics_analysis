
import numpy as np
from typing import Dict, List, Any


class OvarianKnowledgeBase:
    def __init__(self):
        self.marker_genes = self._initialize_comprehensive_marker_genes()
        self.pathways = self._initialize_biological_pathways()
        self.cell_interactions = self._initialize_cell_interactions()
        self.spatial_patterns = self._initialize_spatial_patterns()
        self.developmental_stages = self._initialize_developmental_stages()
        self.functional_modules = self._initialize_functional_modules()

    def _initialize_comprehensive_marker_genes(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            'granulosa_cells': {
                'core': [
                    'Cyp19a1', 'CYP19A1', 'Fshr', 'FSHR', 'Amh', 'AMH', 'Inha', 'INHA',
                    'Inhba', 'INHBA', 'Star', 'STAR', 'Cyp11a1', 'CYP11A1', 'Foxl2', 'FOXL2',
                    'Nr5a1', 'NR5A1', 'Bmp2', 'BMP2', 'Bmp4', 'BMP4', 'Smad3', 'SMAD3'
                ],
                'extended': [
                    'Ar', 'AR', 'Cyp17a1', 'CYP17A1', 'Hsd3b1', 'HSD3B1', 'Runx1', 'RUNX1',
                    'Runx2', 'RUNX2', 'Tgfb1', 'TGFB1', 'Gdf9', 'GDF9', 'Kit', 'KIT',
                    'Kitl', 'KITL', 'Lgr5', 'LGR5', 'Wnt3', 'WNT3', 'Ctnnb1', 'CTNNB1',
                    'Fst', 'FST', 'Acvr1', 'ACVR1', 'Bmpr2', 'BMPR2', 'Smad4', 'SMAD4'
                ],
                'specialized': [
                    'Cyp1b1', 'CYP1B1', 'Hsd17b1', 'HSD17B1', 'Akr1c18', 'AKR1C18',
                    'Ptger2', 'PTGER2', 'Lhx9', 'LHX9', 'Nr0b1', 'NR0B1', 'Gata4', 'GATA4'
                ]
            },
            'theca_cells': {
                'core': [
                    'Cyp17a1', 'CYP17A1', 'Lhcgr', 'LHCGR', 'Insl3', 'INSL3', 'Star', 'STAR',
                    'Hsd3b2', 'HSD3B2', 'Cyp11a1', 'CYP11A1', 'Hsd17b3', 'HSD17B3'
                ],
                'extended': [
                    'Ar', 'AR', 'Akr1c3', 'AKR1C3', 'Srd5a1', 'SRD5A1', 'Ins1', 'INS1',
                    'Ins2', 'INS2', 'Pdgfa', 'PDGFA', 'Vegfa', 'VEGFA', 'Igf1', 'IGF1',
                    'Igfbp2', 'IGFBP2', 'Igfbp5', 'IGFBP5', 'Cebpb', 'CEBPB'
                ],
                'specialized': [
                    'Cyp1a1', 'CYP1A1', 'Akr1c14', 'AKR1C14', 'Ptgs2', 'PTGS2',
                    'Nrg1', 'NRG1', 'Erbb2', 'ERBB2', 'Erbb3', 'ERBB3'
                ]
            },
            'oocytes': {
                'core': [
                    'Zp1', 'ZP1', 'Zp2', 'ZP2', 'Zp3', 'ZP3', 'Zp4', 'ZP4',
                    'Gdf9', 'GDF9', 'Bmp15', 'BMP15', 'Figla', 'FIGLA', 'Nobox', 'NOBOX',
                    'Sohlh1', 'SOHLH1', 'Lhx8', 'LHX8', 'Oosp1', 'OOSP1'
                ],
                'extended': [
                    'Dazl', 'DAZL', 'Ddx4', 'DDX4', 'Dppa3', 'DPPA3', 'Pou5f1', 'POU5F1',
                    'Nanog', 'NANOG', 'Kit', 'KIT', 'Gja4', 'GJA4', 'Mos', 'MOS',
                    'Zar1', 'ZAR1', 'Nlrp5', 'NLRP5', 'Dppa5a', 'DPPA5A'
                ],
                'specialized': [
                    'Scp3', 'SCP3', 'Rec8', 'REC8', 'Dmc1', 'DMC1', 'Sycp1', 'SYCP1',
                    'Msh4', 'MSH4', 'Msh5', 'MSH5', 'Mlh1', 'MLH1'
                ]
            },
            'stromal_fibroblasts': {
                'core': [
                    'Col1a1', 'COL1A1', 'Col3a1', 'COL3A1', 'Vim', 'VIM', 'Fn1', 'FN1',
                    'Acta2', 'ACTA2', 'Thy1', 'THY1', 'Pdgfra', 'PDGFRA', 'Pdgfrb', 'PDGFRB'
                ],
                'extended': [
                    'Col4a1', 'COL4A1', 'Col5a1', 'COL5A1', 'Col6a1', 'COL6A1',
                    'Dcn', 'DCN', 'Bgn', 'BGN', 'Lum', 'LUM', 'Thbs1', 'THBS1',
                    'Sparc', 'SPARC', 'Postn', 'POSTN', 'Fbn1', 'FBN1'
                ],
                'specialized': [
                    'Des', 'DES', 'Myh11', 'MYH11', 'Cnn1', 'CNN1', 'Tagln', 'TAGLN',
                    'Acta1', 'ACTA1', 'Tpm1', 'TPM1', 'Tpm2', 'TPM2'
                ]
            },
            'endothelial_cells': {
                'core': [
                    'Pecam1', 'PECAM1', 'Vwf', 'VWF', 'Eng', 'ENG', 'Cdh5', 'CDH5',
                    'Flt1', 'FLT1', 'Kdr', 'KDR', 'Tie1', 'TIE1', 'Cd31', 'CD31'
                ],
                'extended': [
                    'Nos3', 'NOS3', 'Tek', 'TEK', 'Angpt1', 'ANGPT1', 'Angpt2', 'ANGPT2',
                    'Vegfb', 'VEGFB', 'Fgf2', 'FGF2', 'Pdgfb', 'PDGFB', 'Dll4', 'DLL4',
                    'Hey1', 'HEY1', 'Hey2', 'HEY2', 'Notch4', 'NOTCH4'
                ],
                'specialized': [
                    'Erg', 'ERG', 'Fli1', 'FLI1', 'Sox17', 'SOX17', 'Sox18', 'SOX18',
                    'Etv2', 'ETV2', 'Tal1', 'TAL1', 'Lmo2', 'LMO2'
                ]
            },
            'immune_macrophages': {
                'core': [
                    'Cd68', 'CD68', 'Cd163', 'CD163', 'Csf1r', 'CSF1R', 'Aif1', 'AIF1',
                    'Lyz2', 'LYZ2', 'Cd14', 'CD14', 'Adgre1', 'ADGRE1', 'Itgam', 'ITGAM'
                ],
                'extended': [
                    'Mrc1', 'MRC1', 'Arg1', 'ARG1', 'Il10', 'IL10', 'Vegfa', 'VEGFA',
                    'Mmp9', 'MMP9', 'Mmp2', 'MMP2', 'Ccl2', 'CCL2', 'Il1b', 'IL1B',
                    'Tnf', 'TNF', 'Il6', 'IL6', 'Nos2', 'NOS2'
                ],
                'specialized': [
                    'Marco', 'MARCO', 'Msr1', 'MSR1', 'Cd206', 'CD206', 'Fizz1', 'FIZZ1',
                    'Chi3l3', 'CHI3L3', 'Tgm2', 'TGM2', 'F13a1', 'F13A1'
                ]
            },
            'immune_tcells': {
                'core': [
                    'Cd3e', 'CD3E', 'Cd3d', 'CD3D', 'Cd3g', 'CD3G', 'Cd2', 'CD2',
                    'Ccl5', 'CCL5', 'Il7r', 'IL7R', 'Cd8a', 'CD8A', 'Cd4', 'CD4'
                ],
                'extended': [
                    'Cd28', 'CD28', 'Ctla4', 'CTLA4', 'Pdcd1', 'PDCD1', 'Ifng', 'IFNG',
                    'Il2', 'IL2', 'Il4', 'IL4', 'Foxp3', 'FOXP3', 'Tbx21', 'TBX21',
                    'Gata3', 'GATA3', 'Rorc', 'RORC', 'Il17a', 'IL17A'
                ],
                'specialized': [
                    'Tcf7', 'TCF7', 'Lef1', 'LEF1', 'Sell', 'SELL', 'Ccr7', 'CCR7',
                    'Cd44', 'CD44', 'Cd62l', 'CD62L', 'Klf2', 'KLF2'
                ]
            },
            'surface_epithelium': {
                'core': [
                    'Epcam', 'EPCAM', 'Krt8', 'KRT8', 'Krt18', 'KRT18', 'Krt19', 'KRT19',
                    'Cdh1', 'CDH1', 'Tjp1', 'TJP1', 'Ocln', 'OCLN'
                ],
                'extended': [
                    'Krt7', 'KRT7', 'Krt20', 'KRT20', 'Muc1', 'MUC1', 'Muc4', 'MUC4',
                    'Wt1', 'WT1', 'Msln', 'MSLN', 'Cldn3', 'CLDN3', 'Cldn4', 'CLDN4',
                    'Cldn7', 'CLDN7', 'Dsp', 'DSP', 'Pkp1', 'PKP1'
                ],
                'specialized': [
                    'Pax8', 'PAX8', 'Hoxb13', 'HOXB13', 'Ovgp1', 'OVGP1',
                    'Muc16', 'MUC16', 'Ca125', 'CA125'
                ]
            }
        }

    def _initialize_biological_pathways(self) -> Dict[str, List[str]]:
        return {
            'steroidogenesis': [
                'Star', 'STAR', 'Cyp11a1', 'CYP11A1', 'Hsd3b1', 'HSD3B1', 'Hsd3b2', 'HSD3B2',
                'Cyp17a1', 'CYP17A1', 'Cyp19a1', 'CYP19A1', 'Hsd17b1', 'HSD17B1', 'Hsd17b3', 'HSD17B3',
                'Akr1c3', 'AKR1C3', 'Srd5a1', 'SRD5A1', 'Nr5a1', 'NR5A1'
            ],
            'folliculogenesis': [
                'Foxl2', 'FOXL2', 'Amh', 'AMH', 'Bmp15', 'BMP15', 'Gdf9', 'GDF9',
                'Kit', 'KIT', 'Kitl', 'KITL', 'Fshr', 'FSHR', 'Lhcgr', 'LHCGR',
                'Inha', 'INHA', 'Inhba', 'INHBA', 'Acvr1', 'ACVR1', 'Bmpr2', 'BMPR2'
            ],
            'ovulation': [
                'Ptgs2', 'PTGS2', 'Adamts1', 'ADAMTS1', 'Has2', 'HAS2', 'Tnfaip6', 'TNFAIP6',
                'Areg', 'AREG', 'Ereg', 'EREG', 'Btc', 'BTC', 'Egfr', 'EGFR',
                'Pgr', 'PGR', 'Esr1', 'ESR1', 'Esr2', 'ESR2'
            ],
            'angiogenesis': [
                'Vegfa', 'VEGFA', 'Vegfb', 'VEGFB', 'Vegfc', 'VEGFC', 'Angpt1', 'ANGPT1',
                'Angpt2', 'ANGPT2', 'Pecam1', 'PECAM1', 'Flt1', 'FLT1', 'Kdr', 'KDR',
                'Tie1', 'TIE1', 'Tek', 'TEK', 'Pdgfb', 'PDGFB'
            ],
            'hormone_signaling': [
                'Esr1', 'ESR1', 'Esr2', 'ESR2', 'Pgr', 'PGR', 'Ar', 'AR',
                'Fshr', 'FSHR', 'Lhcgr', 'LHCGR', 'Gnrhr', 'GNRHR', 'Oxtr', 'OXTR',
                'Prlr', 'PRLR', 'Igf1r', 'IGF1R', 'Insr', 'INSR'
            ],
            'immune_response': [
                'Il1b', 'IL1B', 'Il6', 'IL6', 'Tnf', 'TNF', 'Cd68', 'CD68',
                'Cd3e', 'CD3E', 'Ccl2', 'CCL2', 'Ccl5', 'CCL5', 'Cxcl10', 'CXCL10',
                'Ifng', 'IFNG', 'Il10', 'IL10', 'Tgfb1', 'TGFB1'
            ],
            'cell_cycle': [
                'Ccnd1', 'CCND1', 'Ccnd2', 'CCND2', 'Ccne1', 'CCNE1', 'Cdk2', 'CDK2',
                'Cdk4', 'CDK4', 'Rb1', 'RB1', 'E2f1', 'E2F1', 'Tp53', 'TP53',
                'Cdkn1a', 'CDKN1A', 'Cdkn1b', 'CDKN1B', 'Pcna', 'PCNA'
            ],
            'apoptosis': [
                'Bax', 'BAX', 'Bcl2', 'BCL2', 'Tp53', 'TP53', 'Casp3', 'CASP3',
                'Casp8', 'CASP8', 'Casp9', 'CASP9', 'Fas', 'FAS', 'Fasl', 'FASL',
                'Tnfrsf1a', 'TNFRSF1A', 'Apaf1', 'APAF1'
            ]
        }

    def _initialize_cell_interactions(self) -> Dict[str, List[str]]:
        return {
            'granulosa_oocyte': [
                'Gja4', 'GJA4', 'Kit', 'KIT', 'Kitl', 'KITL', 'Gdf9', 'GDF9',
                'Bmp15', 'BMP15', 'Cx37', 'CX37', 'Cx43', 'CX43'
            ],
            'theca_granulosa': [
                'Igf1', 'IGF1', 'Pdgfa', 'PDGFA', 'Vegfa', 'VEGFA', 'Angpt1', 'ANGPT1',
                'Nrg1', 'NRG1', 'Erbb2', 'ERBB2', 'Wnt4', 'WNT4'
            ],
            'immune_stromal': [
                'Csf1', 'CSF1', 'Csf1r', 'CSF1R', 'Ccl2', 'CCL2', 'Cxcl12', 'CXCL12',
                'Il34', 'IL34', 'Pdgfra', 'PDGFRA'
            ],
            'endothelial_support': [
                'Vegfa', 'VEGFA', 'Angpt1', 'ANGPT1', 'Pdgfb', 'PDGFB', 'Notch3', 'NOTCH3',
                'Jag1', 'JAG1', 'Dll4', 'DLL4'
            ]
        }

    def _initialize_spatial_patterns(self) -> Dict[str, Dict[str, Any]]:
        return {
            'follicular_structure': {
                'center': ['oocytes'],
                'inner_layer': ['granulosa_cells'],
                'outer_layer': ['theca_cells'],
                'supporting': ['stromal_fibroblasts']
            },
            'corpus_luteum': {
                'dominant': ['granulosa_cells', 'theca_cells'],
                'supporting': ['endothelial_cells', 'immune_macrophages']
            },
            'stromal_background': {
                'diffuse': ['stromal_fibroblasts', 'immune_macrophages'],
                'vascular': ['endothelial_cells']
            },
            'surface_epithelium_layer': {
                'surface': ['surface_epithelium'],
                'transition': ['stromal_fibroblasts']
            }
        }

    def _initialize_developmental_stages(self) -> Dict[str, List[str]]:
        return {
            'primordial': [
                'Foxl2', 'FOXL2', 'Amh', 'AMH', 'Nobox', 'NOBOX', 'Sohlh1', 'SOHLH1',
                'Lhx8', 'LHX8', 'Figla', 'FIGLA'
            ],
            'primary': [
                'Foxl2', 'FOXL2', 'Fshr', 'FSHR', 'Kit', 'KIT', 'Kitl', 'KITL',
                'Gdf9', 'GDF9', 'Bmp15', 'BMP15'
            ],
            'secondary': [
                'Fshr', 'FSHR', 'Cyp19a1', 'CYP19A1', 'Inha', 'INHA', 'Bmp15', 'BMP15',
                'Amh', 'AMH', 'Inhibin', 'INHIBIN'
            ],
            'antral': [
                'Fshr', 'FSHR', 'Cyp19a1', 'CYP19A1', 'Lhcgr', 'LHCGR', 'Star', 'STAR',
                'Cyp11a1', 'CYP11A1', 'Hsd3b2', 'HSD3B2'
            ],
            'preovulatory': [
                'Lhcgr', 'LHCGR', 'Ptgs2', 'PTGS2', 'Adamts1', 'ADAMTS1', 'Has2', 'HAS2',
                'Areg', 'AREG', 'Ereg', 'EREG'
            ],
            'corpus_luteum': [
                'Star', 'STAR', 'Cyp11a1', 'CYP11A1', 'Hsd3b2', 'HSD3B2', 'Vegfa', 'VEGFA',
                'Angpt1', 'ANGPT1', 'Ptgfr', 'PTGFR'
            ]
        }

    def _initialize_functional_modules(self) -> Dict[str, List[str]]:
        return {
            'extracellular_matrix': [
                'Col1a1', 'COL1A1', 'Col3a1', 'COL3A1', 'Col4a1', 'COL4A1',
                'Fn1', 'FN1', 'Lam1', 'LAM1', 'Hspg2', 'HSPG2'
            ],
            'growth_factors': [
                'Igf1', 'IGF1', 'Igf2', 'IGF2', 'Fgf2', 'FGF2', 'Pdgfa', 'PDGFA',
                'Tgfb1', 'TGFB1', 'Bmp4', 'BMP4', 'Vegfa', 'VEGFA'
            ],
            'transcription_factors': [
                'Foxl2', 'FOXL2', 'Nr5a1', 'NR5A1', 'Gata4', 'GATA4', 'Tbx21', 'TBX21',
                'Runx1', 'RUNX1', 'Sox9', 'SOX9', 'Pax8', 'PAX8'
            ]
        }

    def get_total_marker_count(self) -> int:
        all_markers = set()
        for cell_type_data in self.marker_genes.values():
            for category_data in cell_type_data.values():
                all_markers.update(category_data)
        return len(all_markers)

    def get_pathway_count(self) -> int:
        return len(self.pathways)

    def get_cell_type_count(self) -> int:
        return len(self.marker_genes)