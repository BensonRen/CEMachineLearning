#This function is for testing and single plotting purposes
import plotAnalysis as PA
#PA.plotonerun('results/hyper-swipe-with-BN/20190922_205326')
PA.HeatMapBVL(plot_x_name = 'initial_lr', plot_y_name='train batch size',
                        title = 'Accuracy heatmap for batch size and lr',HeatMap_dir = "results/hyper-swipe-with-BN/",
                        feature_1_name = 'initial_lr', feature_2_name = 'train_batch_size')
#PA.HeatMapBVL(plot_x_name = 'Stopping Loss', plot_y_name='bv_loss',save_name = "HaetMap of stop point.png",
#                        title = 'Tandem spectra reconstrcution Loss comparison vs stopping point ',HeatMap_dir = "../swipe_stop_point_compare",
#                        feature_1_name = 'stop_threshold', feature_2_name = None)#,condense_tuple2len =False)
