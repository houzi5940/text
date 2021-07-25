import pyecharts.options as opts
from pyecharts.charts import Line
x=['3月23号','3月24号','3月25号','3月26号','3月27号','3月28号','3月29号','3月30号']
y1=[20,25,26,26,27,27,29,30]
y2=[17,17,19,20,21,22,23,24]
line=(
    Line()
    .add_xaxis(xaxis_data=x)
    .add_yaxis(series_name="最高温度",y_axis=y1,is_symbol_show=True,color='#000000')
    .add_yaxis(series_name="最低温度",y_axis=y2,color='#66CD00')
    .set_global_opts(title_opts=opts.TitleOpts(title="7日最高最低温度图"))

)
line.render_notebook()