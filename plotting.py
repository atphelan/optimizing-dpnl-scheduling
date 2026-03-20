#%% Cell-wise script execution enabled 
import os
import itertools
from typing import List

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import interp2d
from scipy.integrate import solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression

# Alternative plotting library to matplotlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#%%
# LaTeX formatted axis labels/titles/series names
label_lookup = {
	"k1": r"$\Large{k_1}$",
	"k2": r"$\Large{k_2}$",
	"T_S error": r"$\Large{T_S \text{ error}}$",
	"log T_S error": r"$\Large{log_{10}(T_S) \text{ relative error}}$",
	"tg1": r"$\Large{t_{G_1}}$",
	"ts": r"$\Large{t_S}$",
	"T_S formula": r"$\Large{\text{Inferred }\ T_S\ \text{ (hours)}}$",
	"1/k2": r"$\Large{\frac{1}{k_2}}$",
	"log T_G_1": r"$\Large{log_{10}(T_{G_1})}$",
	"log T_S": r"$\Large{log_{10}(T_{S})}$",
	"log BrdU time": r"$\Large{log_{2}(\text{BrdU time})}$",
	"log Wait time": r"$\Large{log_{2}(\text{Wait time})}$",
	"k2 SNR": r"$\Large{\text{SNR}(k_2)}$",
	"Relative k2 error": r"$\Large{\frac{k_2^{inf} - k_2}{k_2}}$",
	"EdU-BrdU-": r"$\large{EdU^-BrdU^-}$",# \ \text{cells}}$",
	"EdU+BrdU-": r"$\large{EdU^+BrdU^-}$",# \ \text{cells}}$",
	"EdU+BrdU+": r"$\large{EdU^+BrdU^+}$",# \ \text{cells}}$",
	"EdU-BrdU+": r"$\large{EdU^-BrdU^+}$",# \ \text{cells}}$",
	"Mean EdU-BrdU-": r"$\large{\langle N(EdU^-BrdU^-)\rangle \ }$",
	"Mean EdU+BrdU-": r"$\large{\langle N(EdU^+BrdU^-)\rangle \ }$",
	"Mean EdU+BrdU+": r"$\large{\langle N(EdU^+BrdU^+)\rangle \ }$",
	"Mean EdU-BrdU+": r"$\large{\langle N(EdU^-BrdU^+)\rangle \ }$",
	"Std EdU-BrdU-": r"$\large{\sigma_{N(EdU^-BrdU^-)}\ }$",
	"Std EdU+BrdU-": r"$\large{\sigma_{N(EdU^+BrdU^-)}\ }$",
	"Std EdU+BrdU+": r"$\large{\sigma_{N(EdU^+BrdU^+)}\ }$",
	"Std EdU-BrdU+": r"$\large{\sigma_{N(EdU^-BrdU^+)}\ }$",
	"SNR EdU-BrdU-": r"$\large{\text{SNR}(EdU^-BrdU^-)}$",
	"SNR EdU+BrdU-": r"$\large{\text{SNR}(EdU^+BrdU^-)}$",
	"SNR EdU+BrdU+": r"$\large{\text{SNR}(EdU^+BrdU^+)}$",
	"SNR EdU-BrdU+": r"$\large{\text{SNR}(EdU^-BrdU^+)}$",
	"BrdU time": r"$\Large{\textbf{Labelling time difference (h)}}$", # r"$\Large{EdU-BrdU \ \text{time difference }(h)}$",
	"NSR k1": r"$\Large{\text{NSR}(k_1)}$",
	"NSR k2": r"$\Large{\text{NSR}(k_2)}$",
	'NSR k1 & k2 relative to 1h': r"$\Large{\textbf{Relative NSR}}$",
	"Number of cells": r"$\Large{\text{Number of cells}}$",
}
get_cbar_tickspace = lambda l: (max(l)-min(l))/7
caption = "Cycle period: 8.5h, Pulse interval: {}h"
gates = {'EdU+BrdU-': 'dodgerblues', 'EdU+BrdU+': 'darkviolets', 'EdU-BrdU+': 'orangereds', 'EdU-BrdU-': 'Greys'}

#%%
def correlation_plot(df, kind='parameter-wise'):
	'''
	Find correlations between all variables with a Mean and Std, plot as heatmap
	'''
	if kind == 'parameter-wise':
		# cols = [c for c in df.columns if (('Mean' in c or 'Std' in c) and 'k' not in c)]
		cols = [c for c in df.columns if (" EdU" not in c and "Mean cells" in c)]
		cols = [c for c in cols if not ('at end' in c)]
		cols = ['BrdU time'] + cols + ['SNR k1', 'SNR k2', 'Relative error k1', 'Relative error k2']
		cmat = np.corrcoef(df[cols].T)
		ax = sns.heatmap(cmat, center=0)
		ax.set_xticklabels(cols)
		ax.set_yticklabels(cols)
		ax.set_xticks(ax.get_xticks())
		ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
		ax.set_yticks(ax.get_xticks())
		ax.set_yticklabels(ax.get_xticklabels(), rotation=0, ha='right')
		return ax
	elif kind == 'noise-wise':
		df['Ensemble Correlations'] = df['Ensemble Correlations'].apply(np.array)
		cmat = np.mean(df['Ensemble Correlations'], axis=0)
		ax = sns.heatmap(cmat, center=0)
		cols = [c for c in df.columns if ('Mean' in c or 'Std' in c)][:-2]
		ax.set_xticklabels(cols)
		ax.set_yticklabels(cols)
		ax.set_xticks(ax.get_xticks())
		ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
		ax.set_yticks(ax.get_xticks())
		ax.set_yticklabels(ax.get_xticklabels(), rotation=0, ha='right')
		return ax

def snr_peak_scatter(df, y):
	l = len(df['k2'].unique())
	fig, ax = plt.subplots()
	for i, k2 in enumerate(df['k2'].unique()):
		bness = i/l
		series_df = df[df['k2']==k2].groupby('BrdU time').agg({'BrdU time': 'first', y: 'mean'}).reset_index(drop=True)
		pd.DataFrame(series_df.iloc[series_df[y].argmax()]).T.plot('BrdU time', y, kind='scatter', color=(0,1-bness,bness), ax=ax)
	# ax.get_legend().remove()
	return ax

def hex_opt_plot(df, x, y, z, opt_c, opt_v, gridsize=(10, 10), normalise=False, offset=[0,0]):
	fig, ax = plt.subplots()
	sum_df = []
	# v0 = np.array([1, -1, 0])/np.sqrt(2)
	# v1 = np.array([-1, -1, 2])/np.sqrt(6)
	# centre = np.array([8,8,8])

	for gname, g_df in df.groupby([x, y, z]):
		# x_data = np.array([gname[0], gname[1], gname[2]])
		max_opt_v = g_df[opt_v].max()
		best_c = g_df[g_df[opt_v] == max_opt_v][opt_c]
		# sum_df.append({x: v0.dot(x_data - centre), y: v1.dot(x_data - centre), opt_c: best_c, opt_v: max_opt_v})
		sum_df.append({x: gname[0], y: gname[1], z: gname[2], opt_c: best_c, opt_v: max_opt_v})
	supplement = [{k: v + (0.3 if k=='ts' else -0.1 if k=='tg1' else 0) for k, v in d.items()} for d in sum_df] # For plot smoothing
	sum_df = pd.DataFrame(sum_df + supplement)
	if normalise:
		for coord in (x, y, z):
			minc = sum_df[coord].min()
			rangec = sum_df[coord].max() - minc
			sum_df[coord] = (sum_df[coord] - minc)/rangec
	extent = [np.min(sum_df[x])+offset[0], np.max(sum_df[x])+offset[0],
		   np.min(sum_df[y])+offset[1], np.max(sum_df[y])+offset[1]]
	hex = plt.hexbin(sum_df[x], sum_df[y], C=sum_df[opt_c], 
		gridsize=gridsize, extent=extent, reduce_C_function=np.mean)
	cb = fig.colorbar(hex, ax=ax, label='Optimal {} for {}'.format(opt_c, opt_v))
	# plt.scatter(sum_df[x], sum_df[y], marker='.')
	# ax.set_aspect('equal', adjustable='box')
	# ax.set_aspect(1.2)
	ax.set_xlabel(x)
	ax.set_ylabel(y)

from scipy.interpolate import UnivariateSpline

#%%
def peak_splines(df, x, y, grouping, n_groups, colours='Blues', extremum='max'):
	'''
	Plot a set of smooth curves of x against y, with n_groups different curves drawn from data grouped by value of variable 'grouping'.
	'''
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(layout=layout)
	binned_df = df.copy()
	binned_df[grouping] = pd.cut(binned_df[grouping], n_groups)
	# binned_df[grouping] = binned_df[grouping].apply(lambda r: np.mean([r.left, r.right]))
	binned_df['group midpoints'] = binned_df[grouping].apply(lambda r: (r.left+r.right)/2)
	binned_df['group midpoints'] = binned_df['group midpoints'].astype('float64')
	min_g = binned_df['group midpoints'].min()
	max_g = binned_df['group midpoints'].max()
	binned_df['group colour'] = (binned_df['group midpoints'] - min_g)/(max_g - min_g)
	cmap = cm.get_cmap(colours)
	binned_df['group colour'] = binned_df['group colour'].apply(lambda r: cmap(r))
	for gname, gdf in binned_df.groupby([grouping]):
		gcol = np.array(list(gdf['group colour'].unique()[0])[0:3])*230
		gcol = f"rgb({gcol[0]},{gcol[1]},{gcol[2]})"
		gdf = gdf.groupby(x).agg({x: 'first', grouping: 'first', y: 'mean'})
		# gdf = gdf.agg({x: 'first', y: 'mean', grouping: 'first'})
		fig.add_trace(
			go.Scatter(
				x=gdf[x], y=gdf[y],
			  	line_shape='spline', name=str(gname),
			  	marker=dict(size=8, color=gcol, line=dict(width=1.8)),
				line=dict(color=gcol, width=4.0),
				showlegend=False,
			)
		)
	for gname, gdf in binned_df.groupby([grouping]):
		gcol = np.array(list(gdf['group colour'].unique()[0])[0:3])*230
		gcol = f"rgb({gcol[0]},{gcol[1]},{gcol[2]})"
		gdf = gdf.groupby(x).agg({x: 'first', grouping: 'first', y: 'mean'})
		# mark peak
		# --- mark spline extremum ---
		xs = gdf[x].values
		ys = gdf[y].values

		# fit spline through the plotted points
		spline = UnivariateSpline(xs, ys, s=0)

		# dense grid to find true extremum
		x_dense = np.linspace(xs.min(), xs.max(), 500)
		y_dense = spline(x_dense)

		skip_extremum_marker = False
		if extremum == 'max':
			i = np.argmax(y_dense)
		elif extremum == 'min':
			i = np.argmin(y_dense)
		elif extremum == 'none' or extremum is None:
			skip_extremum_marker = True
		else:
			raise ValueError("extremum must be 'max', 'min', or 'none'")

		if not skip_extremum_marker:
			fig.add_trace(go.Scatter(
				x=[x_dense[i]],
				y=[y_dense[i]],
				marker=dict(color=gcol, size=18),
				marker_symbol='star',
				marker_line_width=2,
				showlegend=False
			))

	# To make the colourscale consistent between all curves and only appear once
	fig.add_trace(go.Scatter(
		x=np.array([1,2]),
		y=np.array([binned_df[y].mean()]*2),
		opacity=0,
		marker=dict(
			color=np.array([min_g, max_g]),
			colorbar=dict(
				title=dict(text=grouping, side="top"),
				nticks=8,
				thickness=18
			),
			colorscale=colours,
		),
		showlegend=False,
	))

	# fig.update_traces(
	# 	marker=dict(size=12, line=dict(width=2,)),
	# 	selector=dict(mode='markers')
	# )

	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=11),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=11),#, tickformat='.1e'),
		autosize=False,
		height=400,
		width=600,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=22),
		boxmode='group',
		legend=dict(
			title="Test",
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=3,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=2,
		gridcolor='gray',
		linewidth=1,
		linecolor='black',
		mirror=True,
		range=[1.0, 12.01],
		dtick=1,
	)
	# fig.update_layout(
	# 	xaxis = dict(
	# 		tickmode = 'array',
	# 		tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
	# 		ticktext=['1', '', '3', '', '5', '', '7', '', '9', '', '11', ''],		)
	# )
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=3,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=2,
		gridcolor='gray',
		nticks=8,
		linewidth=1,
		linecolor='black',
		mirror=True,
		# range=[0.0, 0.11],
		range=[15,40.01],
		# showexponent='all',
		# exponentformat='E',
	)

	# fig.update_layout(
	# 	yaxis = dict(
	# 		tickmode = 'array',
	# 		tickvals = [4, 5, 6, 7, 8, 9, 10],
	# 		ticktext=['4', '', '6', '', '8', '', '10'],		)
	# )

	# fig.add_annotation(
	# 	text=r"$\left| \frac{\langle t_{G_1}^{inf}\rangle \ - \  t_{G_1}^{true}}{t_{G_1}^{true}} \right| \ \ $", 
	# 	align='left',
	# 	showarrow=False,
	# 	xref='x domain',
	# 	yref='y domain',
	# 	x=0.95,
	# 	y=0.95,
	# 	bordercolor='black',
	# 	borderwidth=2,
	# 	bgcolor="white",
	# 	font=dict(size=32),
	# 	borderpad=12,
	# )
	
	return fig


#%%
def four_curve_plot(
	df_lines,
	df_points,
	x,
	y,
	loc_vars=['k1', 'k2'],
	pct_window=5,
	err_prop=0.1,
):
	"""
	Plot four curves with proportional error bars and cross markers
	using separate dataframes for lines and points.

	Parameters
	----------
	df_lines : pandas.DataFrame
		Data used to compute spline lines and error bars
	df_points : pandas.DataFrame
		Data used for cross markers
	x : str
		x-axis column
	y : list[str]
		Four dataframe columns defining the curves
	loc_vars : list[str]
		Variables used to filter around their mean
	pct_window : float
		Percent window around mean used to select rows
	err_prop : float
		Error bar size as proportion of y
	"""

	colours = ['dodgerblue', 'orangered', 'darkviolet', 'grey']

	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(layout=layout)

	# --- window filter based on df_lines ---
	m = df_lines[loc_vars].mean().to_numpy()
	window = pct_window / 100

	subset_lines = df_lines.copy()
	subset_points = df_points.copy()

	for i, v in enumerate(loc_vars):
		subset_lines = subset_lines[
			(subset_lines[v] >= m[i] * (1 - window)) &
			(subset_lines[v] <= m[i] * (1 + window))
		]

		subset_points = subset_points[
			(subset_points[v] >= m[i] * (1 - window)) &
			(subset_points[v] <= m[i] * (1 + window))
		]

	# --- draw curves ---
	for var, colour in zip(y, colours):

		gdf_lines = (
			subset_lines
			.groupby(x, as_index=False)[var]
			.mean()
			.sort_values(x)
		)

		gdf_points = (
			subset_points
			.groupby(x, as_index=False)[var]
			.mean()
			.sort_values(x)
		)

		# yerr = gdf_lines[var] * err_prop
		yerr = np.mean(gdf_lines[var])/np.sqrt(gdf_lines[var]) * err_prop

		# spline line + error bars
		fig.add_trace(
			go.Scatter(
				x=gdf_lines[x],
				y=gdf_lines[var],
				mode='lines',
				line_shape='spline',
				name=label_lookup.get(var),
				line=dict(color=colour, width=2,),
				error_y=dict(
					type='data',
					array=yerr,
					# visible=True,
					visible=False,
					width=6,
					color=colour,
				),
			)
		)

		# cross markers from second dataframe
		fig.add_trace(
			go.Scatter(
				x=gdf_points[x],
				y=gdf_points[var],
				mode='markers',
				marker=dict(
					symbol='x-thin',
					size=8,
					color=colour,
					line=dict(width=2, color=colour),
				),
				showlegend=False
			)
		)

	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=11),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=11),#, tickformat='.1e'),
		autosize=False,
		height=300,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=24),
		legend=dict(
			# yanchor="bottom",
			# y=0.05,
			yanchor="top",
			y=0.82,
			xanchor="left",
			x=0.1,
			bordercolor='dimgrey',
			borderwidth=2,
			font=dict(size=18),
			itemwidth=30
		),
	)

	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=3,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=2,
		gridcolor='gray',
		linewidth=1,
		linecolor='black',
		mirror=True,
		dtick=1,
		range=[1,12.01]
	)

	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=3,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=2,
		gridcolor='gray',
		nticks=8,
		linewidth=1,
		linecolor='black',
		mirror=True,
		range = [0,230],
		# range=[0,11],
		# showexponent='all',
		# exponentformat='E',
	)

	return fig

#%%
def peak_splines_three_datasets(
	df_solid,   # 3-stage cycle
	df_dashed,  # Erlang distributed cycle
	df_dotted,  # Noisy initial conditions
	x,
	y,
	colours=('black', 'black', 'black'),
	labels=('Series 1', 'Series 2', 'Series 3'),
):
	"""
	Plot mean y(x) curves from three datasets using solid, dashed, and dotted lines,
	marking the maximum of the spline-interpolated curve.
	"""

	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(layout=layout)

	datasets = [
		(df_solid,  'solid', colours[0], labels[0]),
		(df_dashed, 'dash',  colours[1], labels[1]),
		(df_dotted, 'dot',   colours[2], labels[2]),
	]

	for df, dash_style, colour, label in datasets:
		mean_curve = df.groupby(x)[y].mean().reset_index()

		xs = mean_curve[x].values
		ys = mean_curve[y].values

		# plot spline
		fig.add_trace(
			go.Scatter(
				x=xs,
				y=ys,
				line_shape='spline',
				name=label,
				line=dict(
					color=colour,
					width=4.0,
					dash=dash_style
				),
			)
		)

		# --- spline-based peak ---
		spline = UnivariateSpline(xs, ys, s=0)
		x_dense = np.linspace(xs.min(), xs.max(), 500)
		y_dense = spline(x_dense)

		i_max = np.argmax(y_dense)

		fig.add_trace(
			go.Scatter(
				x=[x_dense[i_max]],
				y=[y_dense[i_max]],
				mode='markers',
				marker=dict(
					symbol='star',
					size=12,
					color=colour,
					line=dict(width=1.5, color='black'),
				),
				showlegend=False,
			)
		)

	fig.update_layout(
		autosize=False,
		height=400,
		width=600,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=20),
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="right",
			x=0.95,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)

	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=3,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=2,
		gridcolor='gray',
		linewidth=1,
		linecolor='black',
		mirror=True,
		dtick=1,
		range=[1.0,12.25]
	)

	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=3,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=2,
		gridcolor='gray',
		linewidth=1,
		linecolor='black',
		mirror=True,
		nticks=8,
		range=[15.0, 55.0]
	)

	return fig


# %%
def opt_dt_vs_x(df, x, grouping, opt_indep, opt_score, colours='Blues', extra_grouping=[]):
	sum_df = []
	# if len(df[grouping]) > 8:
	# 	df[f'Coarse {grouping}'] = 8*(df[grouping]-df[grouping].min())/(df[grouping].max()-df[grouping].min())
	# 	df[f'Coarse {grouping}'] = df[f'Coarse {grouping}'].apply(round)
	# 	grouping = f'Coarse {grouping}'
	for gname, g_df in df.groupby([x, grouping]+extra_grouping):
		# x_data = np.array([gname[0], gname[1], gname[2]])
		max_opt_score = g_df[opt_score].max()
		best_indep = float(g_df[g_df[opt_score] == max_opt_score][opt_indep])
		# sum_df.append({x: v0.dot(x_data - centre), y: v1.dot(x_data - centre), opt_c: best_c, opt_v: max_opt_v})
		if extra_grouping != []:
			sum_df.append({x: gname[0], grouping: gname[1], opt_indep: best_indep, opt_score: max_opt_score, extra_grouping[0]: gname[2]})
		else:
			sum_df.append({x: gname[0], grouping: gname[1], opt_indep: best_indep, opt_score: max_opt_score})
	sum_df = pd.DataFrame(sum_df)
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(data=go.Scatter(x=sum_df[x], y=sum_df[opt_indep], mode='markers',
		marker=dict(
			size=8,
			line=dict(width=0.5, color='black'),
			color=sum_df[grouping],
			colorscale=colours,
			showscale=True,
			colorbar={'title':grouping, 'nticks': 10, 'thickness': 18}
		),
		showlegend=False
	), layout=layout)
	err_size_regr = LinearRegression()
	err_size_res = err_size_regr.fit(np.array(sum_df[x]).reshape(-1,1), np.array(np.log(sum_df[opt_indep])))
	err_fit = err_size_regr.predict(np.array(sum_df[x]).reshape(-1,1))
	print(err_size_res.get_params())
	fig.add_trace(go.Scatter(x=sum_df[x], y=np.exp(err_fit), mode="lines", marker_color = "black", showlegend=False))
	# analytic_guess_x = np.linspace(1.0, 12.0, 30)
	# analytic_guess_y = lambda t: t * np.log((5*(24-t)/(3*0.1646)-3*t)/(1*(24-t)/(3*0.1646)-3*t))
	# fig.add_trace(go.Scatter(
	# 	x=analytic_guess_x, y=[analytic_guess_y(t) for t in analytic_guess_x],
	# 	opacity=0.5,
	# 	marker=dict(
	# 		color='red',
	# 	),
	# 	showlegend=False,
	# 	mode='lines',
	# ))
	# analytic_guess_y2 = lambda ts, tg2m, z: ts * np.log(3*(tg2m-ts*z)/(tg2m-3*ts*z))
	# fig.add_trace(go.Scatter(
	# 	x=sum_df[x], y=[analytic_guess_y2(ts, tg2m, z) for ts, tg2m, z in zip(sum_df[x], sum_df['tg2m'], sum_df['z'])],
	# 	opacity=0.5,
	# 	marker=dict(
	# 		color='red',
	# 	),
	# 	showlegend=False,
	# 	mode='markers',
	# ))
	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=label_lookup.get(x, None), font=dict(size=24)),
		yaxis_title=dict(text=label_lookup.get(opt_indep, None), font=dict(size=24)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# range=[0.0, 12.01],
		# dtick=1,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=8,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 9.5],
	)
	return fig

#%%
def plotly_scatter(df, x, y, c, colourscheme='Viridis', trendline=False):
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='markers',
		marker=dict(
			size=8,
			line=dict(width=0.5, color='darkblue'),
			color=df[c],
			colorscale=colourscheme,
			showscale=True,
			colorbar={'title':'Null', 'nticks': 10, 'thickness': 18}
		)
	), layout=layout)
	if trendline:
		fig.add_trace(

		)
	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=label_lookup.get(None, None), font=dict(size=24)),
		yaxis_title=dict(text=label_lookup.get(y, None), font=dict(size=24)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=10,
		# range=[0.0, 12.01],
		# dtick=20,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 9.5],
	)
	return fig

#%%

def plotly_2lines(df1, x1, y1, df2, x2, y2, colours=('red', 'blue')):
	layout = go.Layout(
		plot_bgcolor='white',
		yaxis2=dict(
			overlaying='y',
			side='right',
			showgrid=False,
			zeroline=False,
			linewidth=1.5,
			linecolor='black',
			mirror=True,
		)
	)

	fig = go.Figure(layout=layout)

	# First line (left y-axis)
	fig.add_trace(
		go.Scatter(
			x=df1[x1],
			y=df1[y1],
			mode='lines',
			name=y1,
			line=dict(color=colours[0], width=3, dash='dot'),
		)
	)

	# Second line (right y-axis)
	fig.add_trace(
		go.Scatter(
			x=df2[x2],
			y=df2[y2],
			mode='lines',
			name=y2,
			# yaxis='y2',
			line=dict(color=colours[1], width=3, dash='dash'),
		)
	)

	fig.update_layout(
		# xaxis_title=dict(text=label_lookup.get(x1, None), font=dict(size=24)),
		yaxis_title=dict(text=y1, font=dict(size=24)),
		yaxis2_title=dict(text=y2, font=dict(size=24)),
		autosize=False,
		height=400,
		width=600,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="right",
			x=0.95,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)

	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=13,
		range=[1.0, 12.01]
	)

	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# range=[0.0, 45.0]
	)

	return fig


#%%
def demo_plot(df, x, y, c, lookup_df, colourscheme='Viridis'):
	'''
	Plot scatterpoints on a graph for x and y, color points by c and show how close they are to the ground truth
	'''
	pass
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='markers',
		marker=dict(
			size=6,
			line=dict(width=0.5, color='darkblue'),
			color=df[c],
			colorscale=colourscheme,
			showscale=True,
			colorbar={'title':c, 'nticks': 5, 'thickness': 14}
		),
		name='Simulated data',
	), layout=layout)

	groundtruth_df = lookup_df.loc[np.logical_and(lookup_df['BrdU time'].isin(df['BrdU time'].unique()), np.logical_and(
		np.isclose(lookup_df['k1'], df['k1'].mean(), atol=1E-7), np.isclose(lookup_df['k2'], df['k2'].mean(), atol=1E-7)))].copy()
	neighbourhood_df = lookup_df.loc[np.logical_and(lookup_df['BrdU time'].isin(df['BrdU time'].unique()), np.logical_and(
		np.isclose(lookup_df['k1'], df['k1'].mean(), rtol=1E-1), np.isclose(lookup_df['k2'], df['k2'].mean(), rtol=10)))].copy()

	fig.add_traces(data=[go.Scatter(x=groundtruth_df[x], y=groundtruth_df[y], mode='markers',
		marker=dict(
			size=12,
			line=dict(width=0.5, color='darkblue'),
			color=groundtruth_df['BrdU time'],
			symbol='x',
			showscale=False,
			colorscale='Greys',
			# colorbar={'title':'BrdU time', 'nticks': 2, 'thickness': 10,}
		),
		name='True mean'),
		go.Scatter(x=neighbourhood_df[x], y=neighbourhood_df[y], mode='markers',
		marker=dict(
			size=2,
			color='black',
			symbol='circle-dot',
		),
		name='Nearby true means'
	)])

	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=label_lookup.get(None, None), font=dict(size=24)),
		yaxis_title=dict(text=label_lookup.get(y, None), font=dict(size=24)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="bottom",
			y=0.05,
			xanchor="right",
			x=0.95,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		nticks=10,
		# range=[0.0, 12.01],
		# dtick=20,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=1,
		# range=[0.0, 45.0],
	)
	return fig


#%%
def plotly_bar(categories: List[str], heights: List[float], colours: List[str]):
	layout = go.Layout(plot_bgcolor='white')
	fig = go.Figure(data=[go.Bar(
		x=categories,
		y=heights,
		marker_color=colours,
		marker_line_width=1.5,
		marker_line_color='black')
	], layout=layout)
	fig.update_layout({'bargap': 0})
	fig.update_layout(
		xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
		xaxis_title=dict(text=None, font=dict(size=24)),
		yaxis_title=dict(text=None, font=dict(size=24)),
		autosize=False,
		height=400,
		width=500,
		margin=dict(l=0, r=0, t=0, b=0),
		hovermode='closest',
		font=dict(size=18),
		boxmode='group',
		legend=dict(
			yanchor="top",
			y=0.95,
			xanchor="left",
			x=0.05,
			bordercolor='dimgrey',
			borderwidth=2,
		),
	)
	fig.update_xaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		linewidth=1.5,
		linecolor='black',
		mirror=True,
	)
	fig.update_yaxes(
		zeroline=True,
		zerolinewidth=2,
		zerolinecolor='black',
		showgrid=True,
		gridwidth=1,
		gridcolor='gray',
		nticks=10,
		linewidth=1.5,
		linecolor='black',
		mirror=True,
		# dtick=20,
		# range=[0.0, 9.5],
	)
	return fig

'''
DATA IMPORT PART
'''
#%%
print('Reading files')
df = pd.read_json('data/DL bootstrap 2025-12-03.json') # Mahalanobis distance used
# print('k1 values: ', df['k1'].unique(), ';', len(df['k1'].unique()), 'unique values of k1.')
# print('k2 values: ', df['k2'].unique(), ';', len(df['k2'].unique()), 'unique values of k2.')
# df['tg1'] = 1/df['k1']
# df['ts'] = 1/df['k2']
# df['tg2m'] = 1/df['k3']
# df['x'] = df['Initial G1']/(df['Initial G1']+df['Initial S']+df['Initial G2M'])
# df['y'] = df['Initial S']/(df['Initial G1']+df['Initial S']+df['Initial G2M'])
# df['z'] = df['Initial G2M']/(df['Initial G1']+df['Initial S']+df['Initial G2M'])
# for k in ['k1', 'k2']:
# 	df[f'Relative error {k}'] = (df[k] - df[f'Mean inferred {k}'])/df[f'Std inferred {k}']
# df['Total cells at end'] = df['Mean EdU+BrdU+']+df['Mean EdU-BrdU+']+df['Mean EdU+BrdU-']+df['Mean EdU-BrdU-']
# df['Labelled fraction'] = 1 - df['Mean EdU-BrdU-']/df['Total cells at end']

#%%
preprint_df = pd.read_json('DPNL inference results.json') # Can take a while for larger datasets
# print('k1 values: ', preprint_df['k1'].unique(), ';', len(preprint_df['k1'].unique()), 'unique values of k1.')
# print('k2 values: ', preprint_df['k2'].unique(), ';', len(preprint_df['k2'].unique()), 'unique values of k2.')
# preprint_df['tg1'] = 1/preprint_df['k1']
# preprint_df['ts'] = 1/preprint_df['k2']
# preprint_df['tg2m'] = 1/preprint_df['k3']
# preprint_df['x'] = preprint_df['Initial G1']/(preprint_df['Initial G1']+preprint_df['Initial S']+preprint_df['Initial G2M'])
# preprint_df['y'] = preprint_df['Initial S']/(preprint_df['Initial G1']+preprint_df['Initial S']+preprint_df['Initial G2M'])
# preprint_df['z'] = preprint_df['Initial G2M']/(preprint_df['Initial G1']+preprint_df['Initial S']+preprint_df['Initial G2M'])
# for k in ['k1', 'k2']:
# 	preprint_df[f'Relative error {k}'] = (preprint_df[k] - preprint_df[f'Mean inferred {k}'])/preprint_df[f'Std inferred {k}']
# preprint_df['Total cells at end'] = preprint_df['Mean EdU+BrdU+']+preprint_df['Mean EdU-BrdU+']+preprint_df['Mean EdU+BrdU-']+preprint_df['Mean EdU-BrdU-']
# preprint_df['Labelled fraction'] = 1 - preprint_df['Mean EdU-BrdU-']/preprint_df['Total cells at end']
lookup_df = pd.read_json('DPNL lookup.json')

'''
PLOTTING BELOW
Either run code from below or type in an interactive window
'''

if __name__ == '__main__':
	fig = four_curve_plot(lookup_df, preprint_df, 'BrdU time', ['Mean EdU+BrdU-', 'Mean EdU-BrdU+', 'Mean EdU+BrdU+', 'Mean EdU-BrdU-'], err_prop=0)
	fig.show()
	# fig = four_curve_plot(lookup_df, preprint_df, 'BrdU time', ['Std EdU+BrdU-', 'Std EdU-BrdU+', 'Std EdU+BrdU+', 'Std EdU-BrdU-'],
	# 	err_prop=0.1)
	# fig.show()
	# preprint_df['Relative absolute error tg1'] = np.abs((1/preprint_df['Mean inferred k1'] - preprint_df['tg1'])/preprint_df['tg1'])
	# preprint_df['Relative absolute error ts'] = np.abs((1/preprint_df['Mean inferred k2'] - preprint_df['ts'])/preprint_df['ts'])
	# fig = peak_splines(preprint_df, 'BrdU time', 'Relative absolute error tg1', 'tg1', 7, colours='Reds', extremum=None)
	# fig.show()
	# fig = peak_splines(preprint_df, 'BrdU time', 'Relative absolute error ts', 'ts', 7, colours='Blues', extremum=None)
	# fig.show()
	exit()
#%%
# correlation_plot(df, 'parameter-wise')
#%%
# centre_df = df[np.logical_and(np.isclose(df['k1'], 1/11, 1E-4), np.isclose(df['k2'], 0.10840, 1E-4))]
# correlation_plot(centre_df, 'noise-wise')

# %%
# snr_peak_scatter(df, 'SNR k2')
# %%
# hex_opt_plot(df, 'tg1', 'ts', 'tg2m', 'BrdU time', 'SNR k2')
# %%
peak_splines(preprint_df, 'BrdU time', 'SNR k2', 'ts', 7, 'Blues', extremum='max')
# peak_splines(df, 'BrdU time', 'SNR k1', 'tg1', 7, 'Reds', extremum='min')
# %%
opt_dt_vs_x(df, x='ts', grouping='tg1', opt_indep='BrdU time', opt_score='SNR k2')
# %%
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(df['tg1'], df['ts'], df['tg2m'], marker='h')
# # ax.scatter(np.array([24,0,0,8]), np.array([0,24,0,8]), np.array([0,0,24,8]), marker='x')
# ax.set_xlabel('tg1')
# ax.set_ylabel('ts')
# ax.set_zlabel('tg2m')
# ax.view_init(elev = 24, azim = 60)
# plt.show()

# %% Variation of purple double-labeled cells with pulse interval
ax = df.plot('BrdU time', 'Mean EdU+BrdU+', kind='scatter')
ax.scatter(df['BrdU time'], 300*df['y']*np.e**(df['BrdU time']*(-df['k2']+ df['z']*df['k3'])), marker='x', color='r', s=8)
# %% Variation of red BrdU single-positive cells with pulse interval
ax = df.plot('BrdU time', 'Mean EdU-BrdU+', kind='scatter')
ax.scatter(df['BrdU time'], 300*df['y']*(1-np.e**(-df['k2']*df['BrdU time']))*np.e**(df['z']*df['k3']*df['BrdU time']), marker='x', color='r', s=8)
# %% Variation of blue EdU single-positive cells with pulse interval
ax = df.plot('BrdU time', 'Mean EdU+BrdU-', kind='scatter')
ax.scatter(df['BrdU time'], 300*df['y']*(1-np.e**(-df['k2']*df['BrdU time']))*np.e**(df['z']*df['k3']*df['BrdU time']), marker='x', color='r', s=8)
# %% Variation of colourless unlabeled cells with pulse interval
ax = df.plot('BrdU time', 'Mean EdU-BrdU-', kind='scatter')
ax.scatter(df['BrdU time'], 300*(df['x']+df['z'])*(1-df['y']*(1-np.e**(-df['k2']*df['BrdU time'])))*np.e**(df['z']*df['k3']*df['BrdU time']), marker='x', color='r', s=8)
# %%
df['Mean product'] = df['Mean EdU+BrdU-']*df['Mean EdU+BrdU+']*df['Mean EdU-BrdU+']
df['Std product'] = df['Std EdU+BrdU-']*df['Std EdU+BrdU+']*df['Std EdU-BrdU+']
df['SNR labelled cells'] = df['Mean product']/df['Std product']
# %%
df_ts8 = df[np.logical_and(np.isclose(df['ts'], 8.0, 1E-1), np.isclose(df['tg1'], 11.0, 1E-1))]
ax = df_ts8.plot('BrdU time', 'Mean product', kind='scatter')
ax = df_ts8.plot('BrdU time', 'Std product', kind='scatter')

# %%
get_timed_dfs = lambda df: [df.loc[df['BrdU time'] == dt] for dt in sorted(df['BrdU time'].unique())]
timed_dfs = get_timed_dfs(df_ts8)
labels = ['EdU-BrdU+', 'EdU+BrdU-', 'EdU+BrdU+', 'EdU-BrdU-']
separations = [{dt: d[f'Mean {label}'].max() - d[f'Mean {label}'].min()
	for dt, d in df_ts8.groupby('BrdU time')} for label in labels]
plt.scatter(range(1,13), [separations[1][float(i)] for i in range(1, 13)])
df_ts8.plot('BrdU time', 'Std EdU-BrdU+', kind='scatter')
df_ts8['SNR lookup scaled'] = np.product([df_ts8['BrdU time'].map(separations[i])/(df_ts8[f'Std {label}']**0) for i, label in enumerate(labels)], axis=0)
ax = df_ts8.plot('BrdU time', 'SNR lookup scaled', kind='scatter')

# %%
ax.scatter(df_ts8['BrdU time'], 0.0001*300*df_ts8['y']*np.exp(df_ts8['BrdU time']*(-df_ts8['k2']+ df_ts8['z']*df_ts8['k3']))*(300*df_ts8['y']*(1-np.e**(-df_ts8['k2']*df_ts8['BrdU time']))*np.e**(df_ts8['z']*df_ts8['k3']*df_ts8['BrdU time']))**2,  marker='x', color='r', s=8)
ax.scatter(df_ts8['BrdU time'], df_ts8['SNR k2'], color='k', s=4)
# %%
dts = np.arange(1,12)
ax = df_ts8.plot('BrdU time', 'SNR k2', kind='scatter', color='y')
df_ts8.plot('BrdU time', 'Mean EdU+BrdU+', kind='scatter', ax=ax, color='m')
df_ts8.plot('BrdU time', 'Mean EdU-BrdU+', kind='scatter', ax=ax, color='r')
df_ts8.plot('BrdU time', 'Mean EdU+BrdU-', kind='scatter', ax=ax, color='c')
df_ts8.plot('BrdU time', 'Mean EdU-BrdU-', kind='scatter', ax=ax, color='k')

# ax.scatter(dts, 40*np.e**(dts*(-0.125+ 0.16*0.2))*((1-np.e**(-0.125*dts))*np.e**(0.16*0.2*dts)), marker='x', color='r', s=8)
# %% Looking for a trend in the SNR of k2 explained by cell cycle parameters
ax = df_ts8.plot('BrdU time', 'SNR k2', kind='scatter')
df_ts8['Guess SNR k2'] = df_ts8['Mean EdU+BrdU+']*df_ts8['Mean EdU+BrdU-']*df_ts8['Mean EdU-BrdU+']*df_ts8['Mean EdU-BrdU-']/(df_ts8['Std EdU+BrdU+']*df_ts8['Std EdU+BrdU-']*df_ts8['Std EdU-BrdU+']*df_ts8['Std EdU-BrdU-'])
df_ts8.plot('BrdU time', 'Guess SNR k2', kind='scatter', ax=ax, color='m')

#%%
df_600 = pd.read_json('data/DL bootstrap 2026-01-21  600init.json')
df_10r = pd.read_json('data/DL bootstrap 2026-01-22  10rep.json')
peak_splines_three_datasets(preprint_df[np.logical_and(np.isclose(preprint_df['k1'], preprint_df['k1'].mean(), rtol=0.05), np.isclose(preprint_df['k2'], preprint_df['k2'].mean(), rtol=0.05))], df_10r, df_600, 'BrdU time', 'SNR k2')

#%%
label_means = [f'Mean {g}' for g in gates]
label_stds = [s.replace('Mean', 'Std') for s in label_means]
def get_central_sensitivity(df, p='tg1', dp=0.03, label_means=label_means, rtolref=0.02, rtolpert=0.08, scale_by_param=True):
	
	tg1_ref = df['tg1'].mean()
	ts_ref = df['ts'].mean()
	
	wait_col = 'BrdU time'
	label_stds = [s.replace('Mean', 'Std') for s in label_means]

	# --- select reference and perturbed rows ---
	ref = df[
		(np.isclose(df['tg1'], tg1_ref, rtol=rtolref)) &
		(np.isclose(df['ts'], ts_ref, rtol=rtolref))
	].copy()

	if ref.empty:
		raise ValueError('No reference mesh points found. Consider a larger rtolref.')
	elif len(ref)>12:
		raise ValueError('Multiple reference mesh points found. Lower rtolref.')

	pert = df[
		(np.isclose(df['tg1'], tg1_ref + (dp if p=='tg1' else 0), rtol=rtolpert)) &
		(np.isclose(df['ts'], ts_ref + (dp if p=='ts' else 0), rtol=rtolpert))
	].copy()

	if pert.empty:
		raise ValueError('No pertubed mesh points found. Consider a larger rtolpert.')
	elif len(pert)>12:
		raise ValueError('Multiple perturbed mesh points found. Lower rtolpert.')


	ref = ref.sort_values(wait_col).set_index(wait_col)
	pert = pert.sort_values(wait_col).set_index(wait_col)

	common_times = ref.index.intersection(pert.index)
	ref = ref.loc[common_times]
	pert = pert.loc[common_times]

	sens = (pert[label_means] - ref[label_means]) / (dp if scale_by_param else 1)
	snr_sens = np.zeros(len(ref))

	for m, s in zip(label_means, label_stds):
		snr_sens += (sens[m] ** 2) / (ref[s] ** 2)

	snr_sens = pd.Series(
		snr_sens,
		index=common_times,
		name='Sensitivity ' + p
	)

	return snr_sens


#%%
