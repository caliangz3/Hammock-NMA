import streamlit as st
import pandas as pd
import numpy as np
import hammock_plot
import io

st.set_page_config(
    page_title = "Hammock Plot for NMA",
    layout = "wide"
)

st.title("Hammock Plot for NMA")

st.markdown(
    """
    Any description?
    """
)

page = st.sidebar.radio(
    "",
    ("Data upload",
    "Snapshot & Simple hammock plots",
    "Frequency-based plots",
    "Metrics-based plots",
    "Top-k plots",
    "Partial ordering plots")
)

########################################
# Function Definition
########################################
def get_matrices(treatment_effect, small_values_good):
    
    # Generate rank matrix
    if small_values_good:
        rank_matrix = treatment_effect.rank(axis = 1, method = "average").astype(int)
    else:
        rank_matrix = treatment_effect.rank(axis = 1, method = "average", ascending=False).astype(int)

    #rank_matrix.to_csv("rank_matrix.csv", index=False)


    # summarize the probability of each treatment attain a specific rank
    rank_prob_table = pd.concat([rank_matrix[col].value_counts(normalize = True).mul(100).rename(col) 
                                for col in rank_matrix.columns],
                                axis=1).fillna(0)
    rank_prob_table = rank_prob_table.sort_index().round(2)
    rank_prob_table.index.name = "Rank"

    #rank_prob_table.to_csv("posterior_prob.csv", index=True)


    # Generate matrix with rows = treatments, columns = Rank
    treatment_matrix = pd.DataFrame([rank_matrix.columns[np.argsort(row)]
                                    for row in rank_matrix.to_numpy()],
                                    columns=range(1, rank_matrix.shape[1]+1))

    #treatment_matrix.to_csv("treatment_matrix.csv", index=False)

    return rank_matrix, rank_prob_table, treatment_matrix



def to_multiline_names(names, row_length=15):
    '''
    Returns a list of modified name from names such that the length of each line of each element in names is no longer than row_length

    to_multipleline_names: List Nat -> List
    '''
    new_names = []

    for name in names:
        if len(name) > row_length:
            chunks = [name[i:i+row_length] for i in range(0, len(name), row_length)]
            name = "\n".join(chunks)
        new_names.append(name)
    
    return new_names

def plot_hammock(data_df, var, value_order, output_path="", **kwargs):
    hammock = hammock_plot.Hammock(data_df=data_df)
    ax = hammock.plot(var=var, value_order=value_order, save_path=str(output_path), **kwargs)
    return ax

def show_hammock(ax, key, filename = "my_plot.png"):
    st.pyplot(ax.figure, width="content")

    buf = io.BytesIO()
    ax.figure.savefig(buf, format = "png", bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="Download PNG",
        data=buf,
        file_name=filename,
        mime="image/png",
        icon=":material/download:",
        use_container_width=True,
        key = key)

def choose_hi_color(highlight_num):
    default_colors = ["#d62728","#1f77b4","#9238e6", "#ff7f0e",
                      "#2ca02c", "#75660F", "#d648ac", "#25e7cd"]

    cols = st.columns(highlight_num)
    highlight_colors = []

    for i in range(highlight_num):
        with cols[i]:
            color = st.color_picker(f"Color {i+1}", value=default_colors[i % len(default_colors)],
                                    key=f"highlight_color_{i}")
            highlight_colors.append(color)
    return highlight_colors

def MCMC_augmentation(treatment_matrix_metrics_hierarchy, sum_col, augmentation_threshold):

    metrics_hierarchy = treatment_matrix_metrics_hierarchy[sum_col]

    metrics_freq = pd.Series(metrics_hierarchy).value_counts()
    metrics_proportion = metrics_freq / sum(metrics_freq)
    adjust_metrics = metrics_proportion[metrics_proportion < augmentation_threshold].index.tolist()

    manual_rows = []
    for metric in adjust_metrics:
        exist_row_index = treatment_matrix_metrics_hierarchy.index[treatment_matrix_metrics_hierarchy["Hierarchy"] == metric].tolist()
        
        n_create = int(len(treatment_matrix_metrics_hierarchy)*augmentation_threshold - len(exist_row_index))
        create = treatment_matrix_metrics_hierarchy.loc[[exist_row_index[0]] * n_create] 

        manual_rows.append(create)

    manual_rows_df = pd.concat(manual_rows, ignore_index=True)
    modified_treatment_matrix_metrics_hierarchy = pd.concat([treatment_matrix_metrics_hierarchy, manual_rows_df],
                                                            ignore_index=True)
    return modified_treatment_matrix_metrics_hierarchy 


###################################################
# 1. Use default dataset or upload your own dataset
###################################################
if "treatment_effect" not in st.session_state:
    st.session_state["treatment_effect"] = None

if "small_values_good" not in st.session_state:
    st.session_state["small_values_good"] = None

if page == "Data upload":
    st.subheader("Choose a dataset")

    use_default = st.button("Use netmeta-Baker2009 (random-effects model) treatment effects")
    uploaded_file = st.file_uploader("Or upload your own treatment effects CSV file", type = ["csv"])

    if use_default:
        st.session_state["treatment_effect"] = pd.read_csv("treatment_effect.csv")
        st.success("Dataset uploaded successfully.")
    elif uploaded_file is not None:
        st.session_state["treatment_effect"] = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully.")
    

    # matrices setup
    small_values_good_choice = st.radio(
        "Smaller treatment effects indicate better treatments",
        ("True", "False")
    )
    st.session_state["small_values_good"] = (small_values_good_choice == "True")

    if st.session_state["treatment_effect"] is not None:
        row_length = 15
        treatment_effect = st.session_state["treatment_effect"].copy()
        small_values_good = st.session_state["small_values_good"]

        with st.spinner("Processing dataset and generating ranking matrices..."):
            treatment_effect.columns = to_multiline_names(treatment_effect.columns, row_length)
            rank_matrix, prob_matrix, treatment_matrix = get_matrices(treatment_effect, small_values_good
        )

        st.session_state["rank_matrix"] = rank_matrix
        st.session_state["prob_matrix"] = prob_matrix
        st.session_state["treatment_matrix"] = treatment_matrix
        st.info("Data processed successfully")
    else:
        st.info("Please use the default dataset or upload a CSV file.")





####################################################
# 2. Snap shot & simple version treatment plot
####################################################
if page == "Snapshot & Simple hammock plots":

    if st.session_state["treatment_effect"] is not None:

        #row_length = 15
        #treatment_effect = st.session_state["treatment_effect"].copy()
        #treatment_effect.columns = to_multiline_names(treatment_effect.columns, row_length)
        #small_values_good = st.session_state["small_values_good"]

        #rank_matrix, prob_matrix, treatment_matrix = get_matrices(treatment_effect, small_values_good)
        rank_matrix = st.session_state["rank_matrix"]
        prob_matrix = st.session_state["prob_matrix"]
        treatment_matrix = st.session_state["treatment_matrix"]

        rank_order = treatment_matrix.columns.tolist()
        p_best_treatment_order = prob_matrix.iloc[0].sort_values(ascending=False).index.tolist()
        value_order_simple_treatment = {k:p_best_treatment_order[::-1] for k in rank_order}

        graph_col, setting_col = st.columns([2,1])

        with setting_col:
            st.subheader("Snapshot plot settings")

            fig_width = st.number_input("Figure width",min_value = 1.0,max_value=100.0,value=20.0, step= 0.01)
            fig_height = st.number_input("Figure height",min_value = 1.0,max_value=100.0,value=10.0, step = 0.01)
            color = st.color_picker("Figure Color",value="#87CEEB")
            font_size = st.number_input("Font size",min_value = 0.0,max_value=100.0,value=8.5,step = 0.01)
            snapshot_uni_vfill = st.number_input("snapshot_uni_vfill",min_value = 0.0,max_value=1.0,value=0.9,step = 0.01)
            snapshot_uni_hfill = st.number_input("snapshot_uni_hfill",min_value = 0.0, max_value=1.0,value=0.85,step = 0.01)
        
        with graph_col:
            st.subheader("Snapshot Plot")

            label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}
            ax_snaphot = plot_hammock(treatment_matrix, var=rank_order, value_order = value_order_simple_treatment,
                                      uni_hfill = snapshot_uni_hfill, uni_vfill = snapshot_uni_vfill, 
                                      default_color = color, label_options = label_option_simple_treatment)
            show_hammock(ax_snaphot, key="snapshot_download")


        st.divider()
        graph_col2, setting_col2 = st.columns([2, 1])
        
        with setting_col2:
            st.subheader("Hammock plot setting")
            
            Hfig_width = st.number_input("Hammock plot width",min_value = 1.0,max_value=100.0,value=20.0, step= 0.01)
            Hfig_height = st.number_input("Hammock plot height",min_value = 1.0,max_value=100.0,value=10.0, step = 0.01)
            Hcolor = st.color_picker("Hammock plot color",value="#87CEEB")
            Hfont_size = st.number_input("Hammock plot font size",min_value = 0.0,max_value=100.0,value=13.0,step = 0.01)
            axis = st.radio("Which on axes?",("Rank", "Treatment"))
            
        with graph_col2:
            st.subheader("Hammock plot")

            if axis == "Rank":
                label_option_simple_treatment = {k: {"fontsize":Hfont_size} for k in rank_order}
                ax_simple_treatment = plot_hammock(treatment_matrix, var=rank_order, value_order = value_order_simple_treatment, 
                                                default_color = Hcolor, label_options = label_option_simple_treatment, 
                                                same_scale=rank_order, width=Hfig_width, height = Hfig_height)
                show_hammock(ax_simple_treatment, key="simpleTreatment_download")

            else:
                value_order_simple_rank = {t: rank_order[::-1] for t in p_best_treatment_order}
                label_option_simple_rank = {t: {"fontsize":Hfont_size} for t in p_best_treatment_order}

                ax_simple_rank = plot_hammock(rank_matrix, var=p_best_treatment_order, value_order = value_order_simple_rank, 
                                            default_color = Hcolor, label_options = label_option_simple_rank, 
                                            same_scale=p_best_treatment_order, width=Hfig_width, height = Hfig_height)
                show_hammock(ax_simple_rank, key="simpleRreatment_download")

    else:
        st.info("Please use the default dataset or upload a CSV file.")





####################################################
# 3. Frequency-based plots
####################################################
if page == "Frequency-based plots":
    if st.session_state["treatment_effect"] is None:
        st.info("Please use the default dataset or upload a CSV file.")
    else:
        rank_matrix = st.session_state["rank_matrix"]
        prob_matrix = st.session_state["prob_matrix"]
        treatment_matrix = st.session_state["treatment_matrix"]
        
        rank_order = treatment_matrix.columns.tolist()
        p_best_treatment_order = prob_matrix.iloc[0].sort_values(ascending=False).index.tolist()
        value_order_simple_treatment = {k:p_best_treatment_order[::-1] for k in rank_order}

        graph_col, setting_col = st.columns([2, 1])

        with setting_col:
            st.subheader("Setting")

            top_fre = st.number_input("Number of most frequent hierarchies to display",
                                      min_value = 1, max_value=100, value=5)
            highlight_top = st.number_input("Number of displayed hierarchies to highlight",
                                            min_value=1, max_value= int(top_fre),
                                            value = 3,
                                            help = "The most probable k hierarhies will be highlighted using distinct color")
            default_color = st.color_picker("Default Color",value="#D0D1D1")
            hi_color = choose_hi_color(highlight_top)
            fig_width = st.number_input("Figure width",min_value = 1.0,max_value=100.0,value=23.0, step= 0.01)
            fig_height = st.number_input("Figure height",min_value = 1.0,max_value=100.0,value=10.0, step = 0.01)
            font_size = st.number_input("Font size",min_value = 0.0,max_value=100.0,value=13.0,step = 0.01)
            axis = st.radio("Which on axes?",("Rank", "Treatment"))
            
        with graph_col:
            
            st.subheader("Highlighting Frequent Paths")

            row_combine = pd.Series(["|".join(treatment_matrix.iloc[idx].tolist()) for idx in treatment_matrix.index])
            rows_freq = row_combine.value_counts()
            top_treatments = rows_freq[:top_fre].index.tolist()

            freq_hierarchy = []

            for i in range(len(row_combine)):
                if row_combine[i] in top_treatments:
                    index = top_treatments.index(row_combine[i])
                    group = index + 1
                    freq_hierarchy.append(str(group))
                else:
                    group = ">"+str(top_fre)
                    freq_hierarchy.append(group)

            treatment_matrix_freq_hierarchy = treatment_matrix.assign(Hierarchy=freq_hierarchy)
            rank_matrix_freq_hierarchy = rank_matrix.assign(Hierarchy=freq_hierarchy)

            hierarchy_order = [">" + str(top_fre)] + list(map(str, range(top_fre, 0, -1)))
            highlight_hierarchy = list(map(str, range(1, highlight_top+1)))

            if axis == "Rank":
                # treatment plot: Hierarchy is the frequency
                label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}
                
                rank_order_top_hier = ["Hierarchy"] + rank_order
                value_order_freq_treatment = value_order_simple_treatment | {"Hierarchy": hierarchy_order}
                label_option_freq_treatment = label_option_simple_treatment | {"Hierarchy": {"fontsize":font_size}}

                ax_freq_treatment = plot_hammock(treatment_matrix_freq_hierarchy, var=rank_order_top_hier, 
                                                value_order=value_order_freq_treatment, default_color=default_color, 
                                                hi_var="Hierarchy", hi_value=highlight_hierarchy, colors=hi_color,
                                                same_scale=rank_order_top_hier[1:],
                                                label_options = label_option_freq_treatment,
                                                width=fig_width, height=fig_height)
                show_hammock(ax_freq_treatment, key="freq_treatment_download")
            else:
                value_order_simple_rank = {t: rank_order[::-1] for t in p_best_treatment_order}
                label_option_simple_rank = {t: {"fontsize":font_size} for t in p_best_treatment_order}

                treatment_order_top_hier = ["Hierarchy"] + p_best_treatment_order
                value_order_freq_rank = value_order_simple_rank | {"Hierarchy": hierarchy_order}
                label_option_freq_rank = label_option_simple_rank|{"Hierarchy": {"fontsize":font_size}}

                ax_freq_rank = plot_hammock(rank_matrix_freq_hierarchy, var=treatment_order_top_hier, 
                                            value_order = value_order_freq_rank, default_color=default_color, 
                                            hi_var="Hierarchy", hi_value=highlight_hierarchy, colors=hi_color, 
                                            same_scale=treatment_order_top_hier[1:], label_options = label_option_freq_rank,
                                            width=fig_width, height=fig_height)
                show_hammock(ax_freq_rank, key="freq_rank_download")





####################################################
#4. Metrics-based plots
####################################################
if page == "Metrics-based plots":
    if st.session_state["treatment_effect"] is None:
        st.info("Please use the default dataset or upload a CSV file.")
    else:
        treatment_effect = st.session_state["treatment_effect"]
        rank_matrix = st.session_state["rank_matrix"]
        prob_matrix = st.session_state["prob_matrix"]
        treatment_matrix = st.session_state["treatment_matrix"]
        small_values_good = st.session_state["small_values_good"]
        
        rank_order = treatment_matrix.columns.tolist()
        p_best_treatment_order = prob_matrix.iloc[0].sort_values(ascending=False).index.tolist()
        value_order_simple_treatment = {k:p_best_treatment_order[::-1] for k in rank_order}


        posterior_mean = rank_matrix.apply(lambda col: col.mean())
        posterior_mean_rank_order = posterior_mean.sort_values().index.tolist()

        posterior_median = rank_matrix.apply(lambda col: col.median())
        posterior_median_rank_order = posterior_median.sort_values().index.tolist()

        theta_hat = treatment_effect.apply(lambda col: col.mean())
        theta_hat_rank_order = theta_hat.sort_values().index.tolist()
        theta_hat_rank_order = to_multiline_names(theta_hat_rank_order, row_length=15)

        row_combine = pd.Series(["|".join(treatment_matrix.iloc[idx].tolist()) for idx in treatment_matrix.index])

        if small_values_good:
            all_rank = pd.DataFrame({"PBV": p_best_treatment_order,
                                     "Median": posterior_median_rank_order,
                                     "EV": theta_hat_rank_order, 
                                     "SUCRA": posterior_mean_rank_order})
        else:
            all_rank = pd.DataFrame({"PBV": p_best_treatment_order[::-1],
                                     "Median": posterior_median_rank_order[::-1],
                                     "EV": theta_hat_rank_order[::-1], 
                                     "SUCRA": posterior_mean_rank_order[::-1]})
            
        rank_pattern = all_rank.apply(lambda col: "|".join(col))

        pattern_sum = {}
        for idx, value in rank_pattern.items():
            if value not in pattern_sum:
                pattern_sum[value] = []
            pattern_sum[value].append(idx)

        for key in pattern_sum.keys():
            group_name = "/".join(pattern_sum[key])
            pattern_sum[key] = group_name

        metrics_hierarchy = []
        for value in row_combine:
            if value in pattern_sum:
                metrics_hierarchy.append(pattern_sum[value])
            else:
                metrics_hierarchy.append("Others")

        treatment_matrix_metrics_hierarchy = treatment_matrix.assign(Hierarchy=metrics_hierarchy)

        metrics_freq = pd.Series(metrics_hierarchy).value_counts()
        metrics_freq_moveOthers = pd.concat([metrics_freq.drop("Others"), metrics_freq.loc[["Others"]]])
        hierarchy_order = metrics_freq_moveOthers.index.tolist()[::-1]
        highlight_num = len(hierarchy_order)-1


        graph_col, setting_col = st.columns([2, 1])

        with setting_col:
            st.subheader("Setting")

            default_color = st.color_picker("Default Color",value="#D0D1D1")
            hi_color = choose_hi_color(highlight_num)
            fig_width = st.number_input("Figure width",min_value = 1.0,max_value=100.0,value=23.0, step= 0.01)
            fig_height = st.number_input("Figure height",min_value = 1.0,max_value=100.0,value=10.0, step = 0.01)
            font_size = st.number_input("Font size",min_value = 0.0,max_value=100.0,value=13.0,step = 0.01)
            augmentation = st.radio("Need augmentation?",("Yes", "No"), index=1)
            augmentation_threshold = 1
            if augmentation == "Yes":
                augmentation_threshold = st.number_input("Augmentation threshold",min_value = 0.0,max_value=1.0,value=0.02,step = 0.001)

        with graph_col:
            
            st.subheader("Highlighting Paths from Different Ranking Metrics")

            # treatment plot: Hierarchy = metrics
            label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}

            metrics_var = treatment_matrix_metrics_hierarchy.columns.tolist()
            metrics_var = metrics_var[-1:]+metrics_var[:-1]

            value_order_metrics_treatment = value_order_simple_treatment | {"Hierarchy": hierarchy_order}
            label_option_metrics_treatment = label_option_simple_treatment | {"Hierarchy": {"fontsize":font_size}}
            highlight_hierarchy_metrics = hierarchy_order[1:]
            hi_color = hi_color[::-1]

            if augmentation == "No":


                ax_metrics_treatment = plot_hammock(treatment_matrix_metrics_hierarchy, var=metrics_var, 
                                                    value_order=value_order_metrics_treatment, default_color=default_color,
                                                    hi_var="Hierarchy", hi_value=highlight_hierarchy_metrics, 
                                                    colors=hi_color,same_scale=metrics_var[1:],
                                                    label_options = label_option_metrics_treatment,
                                                    width=fig_width, height = fig_height)
                show_hammock(ax_metrics_treatment, key="metrics_treatment_download")
            else:
                modified_treatment_matrix_metrics_hierarchy = MCMC_augmentation(treatment_matrix_metrics_hierarchy, sum_col="Hierarchy", 
                                                                                augmentation_threshold = augmentation_threshold)

                ax_modified_metrics_treatment = plot_hammock(modified_treatment_matrix_metrics_hierarchy,
                                                             var=metrics_var, value_order=value_order_metrics_treatment, 
                                                                            default_color=default_color, hi_var="Hierarchy", 
                                                                            hi_value=highlight_hierarchy_metrics, colors=hi_color,
                                                                            same_scale=metrics_var[1:],
                                                                            label_options = label_option_metrics_treatment,
                                                                            width=fig_width, height = fig_height)
                show_hammock(ax_modified_metrics_treatment, key="metrics_treatment_modified_download")
            


####################################################
#5. Top k
####################################################
if page == "Top-k plots":
    if st.session_state["treatment_effect"] is None:
        st.info("Please use the default dataset or upload a CSV file.")
    else:
        rank_matrix = st.session_state["rank_matrix"]
        prob_matrix = st.session_state["prob_matrix"]
        treatment_matrix = st.session_state["treatment_matrix"]
        p_best_treatment_order = prob_matrix.iloc[0].sort_values(ascending=False).index.tolist()

        graph_col, setting_col = st.columns([2, 1])

        with setting_col:
            st.subheader("Setting")

            top_k = st.number_input("Choose K",min_value = 1,max_value=rank_matrix.shape[1],value=3)
            default_color = st.color_picker("Default Color",value="#D0D1D1")
            hi_color = choose_hi_color(top_k)
            fig_width = st.number_input("Figure width",min_value = 1.0,max_value=100.0,value=23.0, step= 0.01)
            fig_height = st.number_input("Figure height",min_value = 1.0,max_value=100.0,value=10.0, step = 0.01)
            font_size = st.number_input("Font size",min_value = 0.0,max_value=100.0,value=13.0,step = 0.01)
        
        with graph_col:
            others_treatment = p_best_treatment_order[top_k:]
