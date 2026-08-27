import streamlit as st
import pandas as pd
import numpy as np
import hammock_plot
import io

st.set_page_config(
    page_title = "Hammock Plot for NMA",
    layout = "wide"
)

page = st.sidebar.radio(
    "",
    ("Data upload",
    "Snapshot & Standard hammock plots",
    "Frequency-based plots",
    "Metrics-based plots",
    "Top-*k* metrics-based plots",
    "Partial ordering plots")
)

################################################################################
# Global constant & Function Definition
################################################################################
row_length = 15


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


def to_multiline_names(names, wrap_mode, row_length):
    '''
    Returns a list of modified name from names such that the length of each line of each element in names is no longer than row_length,
    wrap_mode no equals to No wrapping

    to_multipleline_names: List Nat -> List
    '''
    if wrap_mode == "No wrapping":
        return list(names)
    new_names = []
    if wrap_mode == "Wrap at spaces":
        for name in names:
            new_names.append(name.replace(" ", "\n"))
    else:
        for name in names:
            if len(name) > row_length:
                chunks = [name[i:i+row_length] for i in range(0, len(name), row_length)]
                name = "\n".join(chunks)
            new_names.append(name)
    return new_names


def plot_hammock(data_df, var, value_order, **kwargs):
    hammock = hammock_plot.Hammock(data_df=data_df)
    ax = hammock.plot(var=var, value_order=value_order, 
                      min_bar_height_connectors = 0,
                      **kwargs)
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
        key = key,
        on_click="ignore")


def choose_hi_color(highlight_num):
    default_colors = ["#fdc086",  "#386cb0", "#7fc97f", "#f0027f"]

    cols = st.columns(highlight_num)
    highlight_colors = []

    for i in range(highlight_num):
        with cols[i]:
            if i < len(default_colors):
                color = st.color_picker(f"Color {i+1}", value=default_colors[i],
                                        key=f"highlight_color_{i}")
            else:
                color = st.color_picker(f"Color {i+1}", value="#00ff00", key=f"highlight_color_{i}")
            highlight_colors.append(color)
    return highlight_colors


def MCMC_augmentation(treatment_matrix_metrics_hierarchy, sum_col, augmentation_threshold):

    metrics_hierarchy = treatment_matrix_metrics_hierarchy[sum_col]

    metrics_freq = pd.Series(metrics_hierarchy).value_counts()
    metrics_proportion = metrics_freq / sum(metrics_freq)
    adjust_metrics = metrics_proportion[metrics_proportion < augmentation_threshold].index.tolist()

    if len(adjust_metrics) == 0:
        return treatment_matrix_metrics_hierarchy

    manual_rows = []
    for metric in adjust_metrics:
        exist_row_index = treatment_matrix_metrics_hierarchy.index[treatment_matrix_metrics_hierarchy[sum_col] == metric].tolist()
        
        n_create = int(len(treatment_matrix_metrics_hierarchy)*augmentation_threshold - len(exist_row_index))
        create = treatment_matrix_metrics_hierarchy.loc[[exist_row_index[0]] * n_create] 

        manual_rows.append(create)

    manual_rows_df = pd.concat(manual_rows, ignore_index=True)
    modified_treatment_matrix_metrics_hierarchy = pd.concat([treatment_matrix_metrics_hierarchy, manual_rows_df],
                                                            ignore_index=True)
    return modified_treatment_matrix_metrics_hierarchy 


def selected_treatment_ordering(key):
    choice = st.radio("Order treatments by: ", options=("SUCRA/Mean rank", "EV", "PBV", "Median rank", "Custom order"),
                      key = key,
                      help = "Order treatments from top to bottom when column names represent 'Rank', or from left to right when column names represent 'Treatment', using:\n"
                      "- SUCRA / Mean Rank: Surface under the cumulative ranking curve, or equivalently the mean of the treatment rank distribution\n" 
                      "- EV: Expected treatment effect\n"
                      "- PBV: Probability of being the best\n"
                      "- Median Rank: Median of the treatment rank distribution\n"
                      "- Custom order: Manually specify your own treatment ordering")
    all_rank = st.session_state["all_rank"]
    if choice == "SUCRA/Mean rank":
        return all_rank["SUCRA"]
    elif choice == "EV":
        return all_rank["EV"]
    elif choice == "PBV":
        return all_rank["PBV"]
    elif choice == "Median rank":
        return all_rank["Median"]
    else:
        selected_order = st.multiselect("Select treatments in the desired order", options=st.session_state["treatment_effect"].columns, 
                                        help="Select all treatments in the desired order.",
                                        key = key + "custom_order")
        return selected_order


def treatment_lebel_wrapping_matrices(rank_matrix, treatment_matrix, all_rank, wrap_mode, row_length):
    if wrap_mode == "No wrapping":
        return rank_matrix, treatment_matrix, all_rank
    wrapped_names = dict(zip(rank_matrix.columns,to_multiline_names(rank_matrix, wrap_mode, row_length)))
    treatment_matrix = treatment_matrix.replace(wrapped_names)
    all_rank = all_rank.replace(wrapped_names)
    rank_matrix.columns = to_multiline_names(rank_matrix.columns, wrap_mode, row_length)

    return rank_matrix, treatment_matrix, all_rank


def plot_general_settng(key):

    font_size = st.number_input("Font size",min_value = 0.0,max_value=100.0,value=13.0,step = 0.5,
                                key = key+"_font")
    wrap_mode = st.radio("Treatment Label Wrapping", ("No wrapping","Wrap at spaces", "Fixed character length"),
                            horizontal=True, key = key+"_wrap_mode",
                            help = "For treatment names displayed on the plot, choose no wrapping, wrapping at every space, or wrapping after a fixed number of characters. " \
                            "Wrapping helps to break long treatment names into multiple lines")
    if wrap_mode == "Fixed character length":
        row_length = st.number_input("Treatment label wrap length",min_value = 5,max_value=30,value=15,step = 1,
                                    help="Treatment names longer than this number of characters will wrap onto multiple lines",
                                    key=key+"_rowLength")
    if key == "snapshot" or key == "simple_hammock":
        fig_width = st.number_input("Figure width (inches)",min_value = 1.0,max_value=100.0,value=20.0, step= 0.5,
                                    key = key + "_width")
    else:
        fig_width = st.number_input("Figure width (inches)",min_value = 1.0,max_value=100.0,value=23.0, step= 0.5,
                                    key = key + "_width")
    if key == "snapshot":
        fig_height = st.number_input("Figure height (inches)",min_value = 1.0,max_value=100.0,value=15.0, step = 0.5,
                                    key = key + "_height")
    else:
        fig_height = st.number_input("Figure height (inches)",min_value = 1.0,max_value=100.0,value=10.0, step = 0.5,
                                    key = key + "_height")
    if wrap_mode == "Fixed character length":
        return font_size, wrap_mode, row_length, fig_width, fig_height
    else:
        return font_size, wrap_mode, None, fig_width, fig_height





################################################################################
# Default datasets & Plot Descriptions
################################################################################
dataset_info = {"Baker2009 (Bayesian random-effects model)": "Baker WL, Baker EL, Coleman CI (2009): Pharmacologic Treatments for Chronic Obstructive Pulmonary Disease: A Mixed-Treatment Comparison Meta-analysis. Pharmacotherapy: The Journal of Human Pharmacology and Drug Therapy, 29, 891–905",
                "Boland2003 (Bayesian random-effects model)": "Lu and Ades (2006), Assessing Evidence Inconsistency in Mixed Treatment Comparisons, Journal of the American Statistical Society, 101(474):447-459. [doi:10.1198/016214505000001302] \n\nBoland et al. (2003), Early thrombolysis for the treatment of acute myocardial infarction: a systematic review and economic evaluation, Health Technology Assessment 7(15):1-136. [doi:10.3310/hta7150]",
                "Dogliotti2014 (Bayesian random-effects model)": "Dogliotti A, Paolasso E, Giugliano RP (2014): Current and new oral antithrombotics in non-valvular atrial fibrillation: a network meta-analysis of 79 808 patients. Heart, 100, 396–405",
                "Dong2013 (Frequentist Mantel-Haenszel method)": "Dong Y-H, Lin H-H, Shau W-Y, Wu Y-C, Chang C-H, Lai M-S (2013): Comparative safety of inhaled medications in patients with chronic obstructive pulmonary disease: systematic review and mixed treatment comparison meta-analysis of randomised controlled trials. Thorax, 68, 48–56",
                "Elliott2007 (Bayesian random-effects model)": "Dias S, Welton NJ, Sutton AJ, Ades AE (2011). “NICE DSU technical support document 2: a generalised linear modelling framework for pairwise and network meta-analysis of randomised controlled trials.” National Institute for Health and Clinical Excellence. \n\nElliott WJ, Meyer PM (2007). “Incident diabetes in clinical trials of antihypertensive drugs: a network meta-analysis.” The Lancet, 369(9557), 201–207.",
                "Franchini2012 (Bayesian fixed-effects model)": "Franchini AJ, Dias S, Ades AE, Jansen JP, Welton NJ (2012): Accounting for correlation in network meta-analysis with multi-arm trials. Research Synthesis Methods, 3, 142–60",
                "Fretheim2012 (Frequentist random-effects model)": "Fretheim A, Odgaard-Jensen J, Brørs O, et al. (2012): Comparative effectiveness of antihypertensive medication for primary prevention of cardiovascular disease: Systematic review and multiple treatments meta-analysis. BMC Medicine, 10, 33." ,
                "Gurusamy2011 (Bayesian fixed-effects model)": "Gurusamy KS, Pissanou T, Pikhart H, Vaughan J, Burroughs AK, Davidson BR (2011): Methods to decrease blood loss and transfusion requirements for liver transplantation. Cochrane Database of Systematic Reviews, CD009052",
                "Rochwerg2014 (Frequentist fix-effects model)": "Rochwerg B, Alhazzani W, Sindi A, et al. (2014): Fluid resuscitation in sepsis: A systematic review and network meta-analysis. Annals of Internal Medicine, 161, 347–355."
                }

featured = {"Boland2003 (Bayesian random-effects model)", "Elliott2007 (Bayesian random-effects model)"}

Description = {"Snapshot_rank": "Analogous to a stacked barchart with direct labeling (i.e. placing labels directly next onto plotting elements rather than using a separate legend). Each bar represents a rank, and each segment represents a treatment. The height of a treatment's segment is proportional to the posterior probability that the treatment occupies that rank.",
               "Snapshot_treatment": "Analogous to a stacked barchart with direct labeling (i.e. placing labels directly next onto plotting elements rather than using a separate legend). Each bar represents a treatment, and each segment represents a rank. The height of a rank's segment is proportional to the posterior probability that the treatment occupies that rank.",
               "Standard Hammock_rank": "Extends the stacked barcharts by visualizing pairwise distributions of treatments occupying adjacent ranks, while preserving marginal distribution at each rank. The height of a univariate bar on a rank axis is proportional to the probability that a treatment occupies that rank. The width of a connector is proportional to the joint posterior probability that one treatment occupies one rank and another treatment occupies the adjacent rank. The connectors leaving a univariate bar partition its height, so their probabilities sum to that bar's marginal probability",
               "Standard Hammock_treatment": "Extends the stacked barcharts by visualizing pairwise distributions of ranks for adjacent treatments, while preserving marginal distribution at each treatment. The height of a univariate bar on a treatment axis is proportional to the probability that the treatment occupies that rank. The width of a connector is proportional to the joint posterior probability that one rank is occupied by one treatment and another rank is occupied by the adjacent treatment. The connectors leaving a univariate bar partition its height, so their probabilities sum to that bar's marginal probability",
               "Frequency-based_rank": "Extends the standard hammock plot by highlighting the most frequent treatment hierarchies to reveal their corresponding posterior probabilities and the uncertainty in the joint posterior distribution. The width of each highlighted hierarchy is proportional to the posterior probability of that complete path and remains constant across all axes.",
               "Frequency-based_treatment": "Extends the standard hammock plot by highlighting the most frequent treatment rankings to reveal their corresponding posterior probabilities and the uncertainty in the joint posterior distribution. The width of each highlighted ranking is proportional to the posterior probability of that complete path and remains constant across all axes.",
               "Metrics-based": "Extends the standard hammock plot by highlighting the treatment hierarchies identified by different ranking metrics to assess how well these hierarchies are supported by the posterior distribution and the associated uncertainty. The width of each highlighted path is proportional to the posterior probability of that complete path and remains constant across all axes.",
               "Top-k_rank": "Focuses on the top *k* rank positions and the corresponding treatments identified by a ranking metric. All remaining treatments are grouped as 'Others', which helps to reduce the number of distinct sample paths. Hierarchies identified by ranking metrics are highlighted",
               "Top-k_treatment": "Focuses on the treatments occupying the top *k* rank positions in a metric-based hierarchy. All ranks greater than *k* are grouped as 'Others', which helps to reduce the number of distinct sample paths. Rankings corresponding to the hierarchies identified by ranking metrics are highlighted",
               "Subsetting": "Focuses on a selected subset of treatments and their relative ordering by excluding other treatments, and helps to reduce the number of distinct samples paths. Hierarchies identified by ranking metrics are highlighted." }





###########################################################################################
# 1. Use default dataset or upload your own dataset
###########################################################################################
if "treatment_effect" not in st.session_state:
    st.session_state["treatment_effect"] = None

if "small_values_good" not in st.session_state:
    st.session_state["small_values_good"] = None

if page == "Data upload":
    st.title("Hammock Plots for Network Meta-Analysis (NMA)")

    st.subheader("Choose a data source")

    mode = st.pills("",["Use an example dataset", "Upload my own treatment effects CSV file"], selection_mode="single", label_visibility = "collapsed")

    if mode == "Use an example dataset":
        dataset_choice = st.radio("Available dataset:", options=list(dataset_info), 
                                  format_func = lambda x: f"**{x}**" if x in featured else x, 
                                  index = 4,
                                  horizontal=True)
        index = dataset_choice.find(" ")
        file_name = "Example_dataset/" + dataset_choice[:index] + "_treatment_effect.csv"
        default_dataset= pd.read_csv(file_name)

        st.caption(dataset_info[dataset_choice])
        st.download_button(label="Download sample dataset", data=default_dataset.to_csv(index=False), 
                           file_name="sample_treatment_effect.csv", mime="text/csv",
                           on_click="ignore")
        st.session_state["treatment_effect"] = default_dataset.copy()
        st.success("Dataset uploaded successfully.")

    elif mode == "Upload my own treatment effects CSV file":
        uploaded_file = st.file_uploader("", type = ["csv"], label_visibility = "collapsed")
        if uploaded_file is not None:
        # warning when upload file size > 10MB
            if uploaded_file.size > 10 * 1024 * 1024:
                st.warning("File size is greater than 10MB, data processing and plots generation will be slow.")
            data = pd.read_csv(uploaded_file)
            st.session_state["treatment_effect"] = data
            st.success("Dataset uploaded successfully.")

    # matrices setup
    small_values_good_choice = st.radio(
        "Smaller treatment effects indicate better treatments?",
        ("True", "False"),
        help="Choose 'True' when smaller treatment effects are more desirable, otherwise choose 'False'."
    )
    
    st.session_state["small_values_good"] = (small_values_good_choice == "True")

    if st.session_state["treatment_effect"] is not None:
        st.dataframe(st.session_state["treatment_effect"], use_container_width=True)

    if st.session_state["treatment_effect"] is not None:
        treatment_effect = st.session_state["treatment_effect"].copy()
        small_values_good = st.session_state["small_values_good"]

        with st.spinner("Processing dataset and generating ranking matrices..."):
            rank_matrix, prob_matrix, treatment_matrix = get_matrices(treatment_effect, small_values_good)

            # metrics ranking hierarchies
            rank_order = treatment_matrix.columns.tolist()
            p_best_treatment_order = prob_matrix.iloc[0].sort_values(ascending=False).index.tolist()
            
            posterior_mean = rank_matrix.apply(lambda col: col.mean())
            posterior_mean_rank_order = posterior_mean.sort_values().index.tolist()

            posterior_median = rank_matrix.apply(lambda col: col.median())
            posterior_median_rank_order = posterior_median.sort_values().index.tolist()

            theta_hat = treatment_effect.apply(lambda col: col.mean())
            theta_hat_rank_order = theta_hat.sort_values().index.tolist()

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
            
        st.markdown("##### Treatment hierarchies based on different ranking metrics:")
        st.table(all_rank)

        st.session_state["rank_matrix"] = rank_matrix
        st.session_state["prob_matrix"] = prob_matrix
        st.session_state["treatment_matrix"] = treatment_matrix
        st.session_state["all_rank"] = all_rank
        st.info("Data processed successfully")
    else:
        st.info("Please use **an example** dataset or upload a CSV file.")





############################################################################################
# 2. Snap shot & simple version treatment plot
############################################################################################
if page == "Snapshot & Standard hammock plots":

    st.title("Hammock Plots for NMA")

    if st.session_state["treatment_effect"] is not None:

        graph_col, setting_col = st.columns([2,1])

        with setting_col:
            st.subheader("Snapshot plot settings")
            snapshot_axis = st.radio("Column names represent: ",("Rank", "Treatment"), key = "snapshot_axis")
            chosen_metric_order = selected_treatment_ordering(key = "snapshot_treatment_order")
            font_size, wrap_mode, row_length, fig_width, fig_height = plot_general_settng("snapshot")
            color = st.color_picker("Figure Color",value="#beaed4") ##87CEEB
            snapshot_uni_vfill = st.slider("Univariate bar vertical fill",min_value = 0.0,max_value=1.0,value=0.9,step = 0.01,
                                           help = "Fraction of vertical space that should be populated by data. Adjusts the height of the data points")
            snapshot_uni_hfill = st.slider("Univariate bar horizontal fill",min_value = 0.0, max_value=1.0,value=0.85,step = 0.01,
                                                 help = "Fraction of horizontal space allocated to labels/univ. bars rather than to connecting boxes")
        

        with graph_col:
            st.subheader("Snapshot Plot")
            with st.expander("Description (Click to expand):"):
                if snapshot_axis == "Rank":
                    st.write(Description["Snapshot_rank"])
                else:
                    st.write(Description["Snapshot_treatment"])

            with st.spinner("Processing data and generating plot..."):
                
                #deal with treatment label wrapping
                rank_matrix, treatment_matrix, all_rank = treatment_lebel_wrapping_matrices(rank_matrix = st.session_state["rank_matrix"].copy(), 
                                                                                            treatment_matrix = st.session_state["treatment_matrix"].copy(), 
                                                                                            all_rank = st.session_state["all_rank"].copy(),
                                                                                            wrap_mode= wrap_mode,
                                                                                            row_length = row_length)
                
                treatment_order = to_multiline_names(chosen_metric_order, wrap_mode, row_length)
                
                rank_order = treatment_matrix.columns.tolist()
                value_order_simple_treatment = {k:treatment_order[::-1] for k in rank_order}
                value_order_simple_rank = {t: rank_order[::-1] for t in treatment_order}

                if snapshot_axis == "Rank":
                    label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}
                    ax_snaphot = plot_hammock(treatment_matrix, var=rank_order, value_order = value_order_simple_treatment,
                                            uni_hfill = snapshot_uni_hfill, uni_vfill = snapshot_uni_vfill, 
                                            default_color = color, label_options = label_option_simple_treatment,
                                            width = fig_width, height = fig_height)
                    show_hammock(ax_snaphot, key="snapshot_download")
                else:
                    label_option_simple_rank = {t: {"fontsize":font_size} for t in treatment_order}
                    ax_snapshot_rank = plot_hammock(rank_matrix, var=treatment_order, value_order = value_order_simple_rank, 
                                                    uni_hfill = snapshot_uni_hfill, uni_vfill = snapshot_uni_vfill, 
                                                    default_color = color, label_options = label_option_simple_rank,
                                                    width = fig_width, height = fig_height)
                    show_hammock(ax_snapshot_rank, key="snapshot_rank_download")

        st.divider()
        graph_col2, setting_col2 = st.columns([2, 1])

        
        with setting_col2:
            st.subheader("Hammock plot setting")
            
            axis = st.radio("Column names represent:",("Rank", "Treatment"), key = "simple_axis")
            chosen_metric_order = selected_treatment_ordering(key = "simple_treatment_order")
            Hfont_size, wrap_mode, row_length, Hfig_width, Hfig_height = plot_general_settng(key = "simple_hammock")
            Hcolor = st.color_picker("Hammock plot color",value="#beaed4") #87CEEB

        with graph_col2:
            st.subheader("Hammock plot")
            with st.expander("Description (Click to expand):"):
                if axis == "Rank":
                    st.write(Description["Standard Hammock_rank"])
                else:
                    st.write(Description["Standard Hammock_treatment"])

            with st.spinner("Processing data and generating plot..."):


                rank_matrix, treatment_matrix, all_rank = treatment_lebel_wrapping_matrices(rank_matrix = st.session_state["rank_matrix"].copy(), 
                                                                                            treatment_matrix = st.session_state["treatment_matrix"].copy(), 
                                                                                            all_rank = st.session_state["all_rank"].copy(),
                                                                                            wrap_mode = wrap_mode,
                                                                                            row_length = row_length)

                treatment_order = to_multiline_names(chosen_metric_order, wrap_mode, row_length)
                value_order_simple_treatment = {k:treatment_order[::-1] for k in rank_order}
                value_order_simple_rank = {t: rank_order[::-1] for t in treatment_order}

                if axis == "Rank":
                    label_option_simple_treatment = {k: {"fontsize":Hfont_size} for k in rank_order}
                    ax_simple_treatment = plot_hammock(treatment_matrix, var=rank_order, value_order = value_order_simple_treatment, 
                                                    default_color = Hcolor, label_options = label_option_simple_treatment, 
                                                    same_scale=rank_order, width=Hfig_width, height = Hfig_height)
                    show_hammock(ax_simple_treatment, key="simpleTreatment_download")

                else:
                    label_option_simple_rank = {t: {"fontsize":Hfont_size} for t in treatment_order}
                    ax_simple_rank = plot_hammock(rank_matrix, var=treatment_order, value_order = value_order_simple_rank, 
                                                default_color = Hcolor, label_options = label_option_simple_rank, 
                                                same_scale=treatment_order, width=Hfig_width, height = Hfig_height)
                    show_hammock(ax_simple_rank, key="simpleRreatment_download")

    else:
        st.info("Please use **an example** dataset or upload a CSV file.")





############################################################################################
# 3. Frequency-based plots
############################################################################################
if page == "Frequency-based plots":

    st.title("Hammock Plots for NMA")

    if st.session_state["treatment_effect"] is None:
        st.info("Please use **an example** dataset or upload a CSV file.")
    else:

        graph_col, setting_col = st.columns([2, 1])

        with setting_col:
            st.subheader("Setting")

            axis = st.radio("Column names represent:",("Rank", "Treatment"), key = "frequency_axis")
            chosen_metric_order = selected_treatment_ordering(key = "frequency_metrics")
            top_fre = st.number_input("Number of most frequent hierarchies to display",
                                      min_value = 1, max_value=100, value=5)
            highlight_top = st.number_input("Number of displayed hierarchies to highlight",
                                            min_value=1, max_value= int(top_fre),
                                            value = 3,
                                            help = "The most probable k hierarchies will be highlighted using distinct colors")
            default_color = st.color_picker("Default Color",value="#D9D9D9")
            hi_color = choose_hi_color(highlight_top)
            font_size, wrap_mode, row_length, fig_width, fig_height = plot_general_settng("Frequency-based")
            augmentation = st.radio("Needs augmentation?",("Yes", "No"), index=1, key="augmentation_frequency")
            augmentation_threshold = 0
            if augmentation == "Yes":
                augmentation_threshold = st.number_input("Augmentation threshold",min_value = 0.0,max_value=1.0,value=0.03,step = 0.01,
                                                         help="Range from 0 to 1. This ensures that the frequency of each selected " \
                                                         "hierarchy is at least (threshold * 100)% of the original sample size.")
        

        with graph_col:
            
            st.subheader("Highlighting Frequent Paths")
            with st.expander("Description (Click to expand):"):
                if axis == "Rank":
                    st.write(Description["Frequency-based_rank"])
                else:
                    st.write(Description["Frequency-based_treatment"])
            
            with st.spinner("Processing data and generating plot..."):

                rank_matrix, treatment_matrix, all_rank = treatment_lebel_wrapping_matrices(rank_matrix = st.session_state["rank_matrix"].copy(), 
                                                                                            treatment_matrix = st.session_state["treatment_matrix"].copy(), 
                                                                                            all_rank = st.session_state["all_rank"].copy(),
                                                                                            wrap_mode = wrap_mode,
                                                                                            row_length = row_length)

                rank_order = treatment_matrix.columns.tolist()
                treatment_order = to_multiline_names(chosen_metric_order, wrap_mode = wrap_mode, row_length=row_length)
                value_order_simple_treatment = {k:treatment_order[::-1] for k in rank_order}

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

                    if augmentation == "No":
                        ax_freq_treatment = plot_hammock(treatment_matrix_freq_hierarchy, var=rank_order_top_hier, 
                                                        value_order=value_order_freq_treatment, default_color=default_color, 
                                                        hi_var="Hierarchy", hi_value=highlight_hierarchy, colors=hi_color,
                                                        same_scale=rank_order_top_hier[1:],
                                                        label_options = label_option_freq_treatment,
                                                        width=fig_width, height=fig_height)
                        show_hammock(ax_freq_treatment, key="freq_treatment_download")

                    else:
                        modified_treatment_matrix_freq_hierarchy = MCMC_augmentation(treatment_matrix_freq_hierarchy, sum_col="Hierarchy", 
                                                                                    augmentation_threshold = augmentation_threshold)

                        ax_freq_treatment_modified = plot_hammock(modified_treatment_matrix_freq_hierarchy,
                                                                var=rank_order_top_hier, value_order=value_order_freq_treatment, 
                                                                default_color=default_color, hi_var="Hierarchy", 
                                                                hi_value=highlight_hierarchy, colors=hi_color,
                                                                same_scale=rank_order_top_hier[1:],
                                                                label_options = label_option_freq_treatment,
                                                                width=fig_width, height = fig_height)
                        show_hammock(ax_freq_treatment_modified, key="freq_treatment_modified_download")

                else:
                    value_order_simple_rank = {t: rank_order[::-1] for t in treatment_order}
                    label_option_simple_rank = {t: {"fontsize":font_size} for t in treatment_order}

                    treatment_order_top_hier = ["Hierarchy"] + treatment_order
                    value_order_freq_rank = value_order_simple_rank | {"Hierarchy": hierarchy_order}
                    label_option_freq_rank = label_option_simple_rank|{"Hierarchy": {"fontsize":font_size}}

                    if augmentation == "No":
                        ax_freq_rank = plot_hammock(rank_matrix_freq_hierarchy, var=treatment_order_top_hier, 
                                                    value_order = value_order_freq_rank, default_color=default_color, 
                                                    hi_var="Hierarchy", hi_value=highlight_hierarchy, colors=hi_color, 
                                                    same_scale=treatment_order_top_hier[1:], label_options = label_option_freq_rank,
                                                    width=fig_width, height=fig_height)
                        show_hammock(ax_freq_rank, key="freq_rank_download")
                    else:
                        modified_rank_matrix_freq_hierarchy = MCMC_augmentation(rank_matrix_freq_hierarchy, sum_col="Hierarchy", 
                                                                                augmentation_threshold = augmentation_threshold)
                        ax_freq_rank_modified = plot_hammock(modified_rank_matrix_freq_hierarchy, var=treatment_order_top_hier, 
                                                            value_order = value_order_freq_rank, default_color=default_color, 
                                                            hi_var="Hierarchy", hi_value=highlight_hierarchy, colors=hi_color, 
                                                            same_scale=treatment_order_top_hier[1:], label_options = label_option_freq_rank,
                                                            width=fig_width, height=fig_height)
                        show_hammock(ax_freq_rank_modified, key="freq_rank_modified_download")




############################################################################################
#4. Metrics-based plots
############################################################################################
if page == "Metrics-based plots":

    st.title("Hammock Plots for NMA")

    graph_col, setting_col = st.columns([2, 1])
    with setting_col:
        st.subheader("Setting")

    if st.session_state["treatment_effect"] is None:
        st.info("Please use **an example** dataset or upload a CSV file.")
    else:
        with graph_col:
            st.subheader("Highlighting Paths from Different Ranking Metrics")
            with st.expander("Description (Click to expand):"):
                st.write(Description["Metrics-based"])

            with st.spinner("Processing data and generating plot..."):

                rank_matrix = st.session_state["rank_matrix"]
                treatment_matrix = st.session_state["treatment_matrix"]
                all_rank = st.session_state["all_rank"]
                
                row_combine = pd.Series(["|".join(treatment_matrix.iloc[idx].tolist()) for idx in treatment_matrix.index])

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


                metrics_freq = pd.Series(metrics_hierarchy).value_counts()
                metrics_freq_moveOthers = pd.concat([metrics_freq.drop("Others"), metrics_freq.loc[["Others"]]])
                hierarchy_order = metrics_freq_moveOthers.index.tolist()[::-1]
                highlight_num = len(hierarchy_order)-1

                with setting_col:

                    chosen_metric_order = selected_treatment_ordering(key = "metric_metrics")
                    default_color = st.color_picker("Default Color",value="#D9D9D9")
                    hi_color = choose_hi_color(highlight_num)
                    font_size, wrap_mode, row_length, fig_width, fig_height = plot_general_settng("Frequency-based")

                    augmentation = st.radio("Needs augmentation?",("Yes", "No"), index=1, key="augmentation_metrics")
                    augmentation_threshold = 0
                    if augmentation == "Yes":
                        augmentation_threshold = st.number_input("Augmentation threshold",min_value = 0.0,max_value=1.0,value=0.03,step = 0.01,
                                                                help="Range from 0 to 1. This ensures that the frequency of each selected " \
                                                                "hierarchy is at least (threshold * 100)% of the original sample size.")


                rank_order = treatment_matrix.columns.tolist()
                treatment_order = to_multiline_names(chosen_metric_order, wrap_mode = wrap_mode, row_length=row_length)
                value_order_simple_treatment = {k:treatment_order[::-1] for k in rank_order}

                with graph_col:

                    rank_matrix, treatment_matrix, all_rank = treatment_lebel_wrapping_matrices(rank_matrix = st.session_state["rank_matrix"].copy(), 
                                                                                                treatment_matrix = st.session_state["treatment_matrix"].copy(), 
                                                                                                all_rank = st.session_state["all_rank"].copy(),
                                                                                                wrap_mode = wrap_mode,
                                                                                                row_length = row_length)
                    treatment_matrix_metrics_hierarchy = treatment_matrix.assign(Metrics=metrics_hierarchy)
                    

                    # treatment plot: Hierarchy = metrics
                    label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}

                    metrics_var = treatment_matrix_metrics_hierarchy.columns.tolist()
                    metrics_var = metrics_var[-1:]+metrics_var[:-1]

                    value_order_metrics_treatment = value_order_simple_treatment | {"Metrics": hierarchy_order}
                    label_option_metrics_treatment = label_option_simple_treatment | {"Metrics": {"fontsize":font_size}}
                    highlight_hierarchy_metrics = hierarchy_order[1:]
                    hi_color = hi_color[::-1]

                    if augmentation == "No":

                        ax_metrics_treatment = plot_hammock(treatment_matrix_metrics_hierarchy, var=metrics_var, 
                                                            value_order=value_order_metrics_treatment, default_color=default_color,
                                                            hi_var="Metrics", hi_value=highlight_hierarchy_metrics, 
                                                            colors=hi_color,same_scale=metrics_var[1:],
                                                            label_options = label_option_metrics_treatment,
                                                            width=fig_width, height = fig_height)
                        show_hammock(ax_metrics_treatment, key="metrics_treatment_download")
                    else:
                        modified_treatment_matrix_metrics_hierarchy = MCMC_augmentation(treatment_matrix_metrics_hierarchy, sum_col="Metrics", 
                                                                                        augmentation_threshold = augmentation_threshold)

                        ax_modified_metrics_treatment = plot_hammock(modified_treatment_matrix_metrics_hierarchy,
                                                                    var=metrics_var, value_order=value_order_metrics_treatment, 
                                                                    default_color=default_color, hi_var="Metrics", 
                                                                    hi_value=highlight_hierarchy_metrics, colors=hi_color,
                                                                    same_scale=metrics_var[1:],
                                                                    label_options = label_option_metrics_treatment,
                                                                    width=fig_width, height = fig_height)
                        show_hammock(ax_modified_metrics_treatment, key="metrics_treatment_modified_download")





############################################################################################
#5. Top k
############################################################################################
if page == "Top-*k* metrics-based plots":

    st.title("Hammock Plots for NMA")

    if st.session_state["treatment_effect"] is None:
        st.info("Please use **an example** dataset or upload a CSV file.")
    else:

        graph_col, setting_col = st.columns([2, 1])
        with setting_col:
            st.subheader("Setting")
            axis = st.radio("Column names represent:",("Rank", "Treatment"), key = "top_k_axis")

        with graph_col:
            st.subheader("Subsetting Based on the Top-*k* Treatments in Metric-Based Hierarchies")
            with st.expander("Description (Click to expand):"):
                if axis == "Rank":
                    st.write(Description["Top-k_rank"])
                else:
                    st.write(Description["Top-k_treatment"])
        
            with st.spinner("Processing data and generating plot..."):

                all_rank = st.session_state["all_rank"].copy()

                with setting_col:

                    chosen_metric_order = selected_treatment_ordering(key = "topk_metrics")
                    top_k = st.number_input("Choose *k*",min_value = 1,max_value=all_rank.shape[0],value=3)
                    font_size, wrap_mode, row_length, fig_width, fig_height = plot_general_settng(key = "topk")
                    default_color = st.color_picker("Default Color",value="#D9D9D9")
                

                rank_matrix, treatment_matrix, all_rank = treatment_lebel_wrapping_matrices(rank_matrix = st.session_state["rank_matrix"].copy(), 
                                                                                            treatment_matrix = st.session_state["treatment_matrix"].copy(), 
                                                                                            all_rank = st.session_state["all_rank"].copy(),
                                                                                            wrap_mode = wrap_mode,
                                                                                            row_length = row_length)

                treatment_order = to_multiline_names(chosen_metric_order, wrap_mode = wrap_mode, row_length=row_length)
                rank_order = treatment_matrix.columns.tolist()


                others_treatment = treatment_order[top_k:]
                

                top_k_treatment = treatment_matrix.copy()
                for i in range(top_k_treatment.shape[0]):
                    current_row = top_k_treatment.iloc[i].isin(others_treatment).to_numpy()
                    top_k_treatment.iloc[i, current_row] = "Others"

                top_k_all_rank= all_rank.copy()
                for j in range(top_k_all_rank.shape[1]):
                    current_column = top_k_all_rank.iloc[:, j].isin(others_treatment)
                    top_k_all_rank.iloc[current_column, j] = "Others"

                row_combine_top_k = pd.Series(["|".join(top_k_treatment.iloc[idx].tolist()) for idx in top_k_treatment.index])

                rank_pattern = top_k_all_rank.apply(lambda col: "|".join(col))

                pattern_sum = {}
                for idx, value in rank_pattern.items():
                    if value not in pattern_sum:
                        pattern_sum[value] = []
                    pattern_sum[value].append(idx)

                for key in pattern_sum.keys():
                    group_name = "/".join(pattern_sum[key])
                    pattern_sum[key] = group_name

                metrics_hierarchy = []
                for value in row_combine_top_k:
                    if value in pattern_sum:
                        metrics_hierarchy.append(pattern_sum[value])
                    else:
                        metrics_hierarchy.append("Others")

                top_k_treatment_matrix_metrics = top_k_treatment.assign(Metrics=metrics_hierarchy)

                # top k treatment plot
                top_var = ["Metrics"] + top_k_treatment_matrix_metrics.columns.tolist()[:top_k]

                metrics_freq = pd.Series(metrics_hierarchy).value_counts()
                metrics_freq_moveOthers = pd.concat([metrics_freq.drop("Others"), metrics_freq.loc[["Others"]]])
                hierarchy_order = metrics_freq_moveOthers.index.tolist()[::-1]
                highlight_hierarchy_metrics = hierarchy_order[1:]
                highlight_num = len(hierarchy_order)-1

                with setting_col:
                    hi_color = choose_hi_color(highlight_num)

                with graph_col:
                    
                    if axis == "Rank":
                        label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}
                        value_order_metrics_treatment = {k:["Others"]+treatment_order[:top_k][::-1] for k in top_k_treatment_matrix_metrics.columns.tolist()[:top_k]} | {"Metrics": hierarchy_order}
                        label_option_metrics_treatment = label_option_simple_treatment | {"Metrics": {"fontsize":font_size}}
                        
                        ax_metrics_topK_treatment = plot_hammock(top_k_treatment_matrix_metrics, var=top_var, 
                                                                value_order=value_order_metrics_treatment, default_color=default_color,
                                                                hi_var="Metrics", hi_value=highlight_hierarchy_metrics, 
                                                                colors=hi_color[::-1],same_scale=top_var[1:],
                                                                label_options = label_option_metrics_treatment,
                                                                width=fig_width, height = fig_height)
                        
                        show_hammock(ax_metrics_topK_treatment, key="ax_metrics_topK_treatment_download")

                    else:
                        top_k_rank = rank_matrix.copy().astype(object)
                        others_treatment = list(range(top_k+1, top_k_rank.shape[1]+1))

                        for i in range(top_k_rank.shape[0]):
                            col = top_k_rank.iloc[i] > top_k
                            top_k_rank.iloc[i, col.values] = "Others"

                        top_k_rank_matrix_metrics = top_k_rank.assign(Metrics=metrics_hierarchy)

                        treatment_order_topk_hier = ["Metrics"] + treatment_order[:top_k]
                        value_order_top_rank = {t: ["Others"]+list(range(1, top_k+1))[::-1] for t in treatment_order[:top_k]} | {"Metrics": hierarchy_order}
                        label_option_simple_rank = {t: {"fontsize":font_size} for t in treatment_order}
                        label_option_top_rank = label_option_simple_rank|{"Metrics": {"fontsize":font_size}}

                        ax_metrics_topK_rank = plot_hammock(top_k_rank_matrix_metrics, var=treatment_order_topk_hier, 
                                                                value_order=value_order_top_rank, default_color=default_color,
                                                                hi_var="Metrics", hi_value=highlight_hierarchy_metrics, 
                                                                colors=hi_color[::-1],same_scale=treatment_order_topk_hier[1:],
                                                                label_options = label_option_top_rank,
                                                                width=fig_width, height = fig_height)
                        show_hammock(ax_metrics_topK_rank, key="metrics_topK_rank_download")





############################################################################################
#6. Subordering
############################################################################################
if page == "Partial ordering plots":

    st.title("Hammock Plots for NMA")

    if st.session_state["treatment_effect"] is None:
        st.info("Please use **an example** dataset or upload a CSV file.")
    else:
        treatment_effect = st.session_state["treatment_effect"]
        all_treatment = list(treatment_effect.columns)

        graph_col, setting_col = st.columns([2, 1])

        with setting_col:
            st.subheader("Setting")
            treatment_subset = st.multiselect("Choose at least **two** treatments to display in order:", options = all_treatment)
            font_size, wrap_mode, row_length, fig_width, fig_height = plot_general_settng(key = "partial_ordering")

        with graph_col:

            st.subheader("Partial ordering of treatments")
            with st.expander("Description (Click to expand):"):
                st.write(Description["Subsetting"])

            selection_message = st.empty()
            if not treatment_subset:
                selection_message.info("Plot will display after choosing a subset of treatments.")
            else:
                selection_message.empty()


            with st.spinner("Processing data and generating plot..."):

                if treatment_subset != []:
                    
                    rank_matrix, treatment_matrix, all_rank = treatment_lebel_wrapping_matrices(rank_matrix = st.session_state["rank_matrix"].copy(), 
                                                                                                treatment_matrix = st.session_state["treatment_matrix"].copy(), 
                                                                                                all_rank = st.session_state["all_rank"].copy(),
                                                                                                wrap_mode = wrap_mode,
                                                                                                row_length = row_length)
                    sub_treatmenet_effect = treatment_effect[treatment_subset]
                    sub_treatmenet_effect.columns = to_multiline_names(sub_treatmenet_effect.columns, wrap_mode = wrap_mode, row_length = row_length)
                    small_values_good = st.session_state["small_values_good"]

                    if small_values_good:
                        sub_rank_matrix = sub_treatmenet_effect.rank(axis = 1, method = "average").astype(int)
                    else:
                        sub_rank_matrix = sub_treatmenet_effect.rank(axis = 1, method = "average", ascending=False).astype(int)

                    sub_treatment_matrix = pd.DataFrame([sub_rank_matrix.columns[np.argsort(row)]
                                                        for row in sub_rank_matrix.to_numpy()],
                                                        columns=range(1, sub_rank_matrix.shape[1]+1))
                    
                    col_name = sub_treatmenet_effect.columns.tolist()
                    sub_rank = all_rank.where(all_rank.isin(col_name))
                    sub_rank = {col: sub_rank[col].dropna().tolist() for col in sub_rank.columns}
                    sub_rank_matrix = pd.DataFrame(sub_rank)

                    rank_pattern = sub_rank_matrix.apply(lambda col: "|".join(col))

                    pattern_sum = {}
                    for idx, value in rank_pattern.items():
                        if value not in pattern_sum:
                            pattern_sum[value] = []
                        pattern_sum[value].append(idx)
                    
                    for key in pattern_sum.keys():
                        group_name = "/".join(pattern_sum[key])
                        pattern_sum[key] = group_name
                    
                    row_combine_subset = pd.Series(["|".join(sub_treatment_matrix.iloc[idx].tolist()) for idx in sub_treatment_matrix.index])
                    
                    metrics_hierarchy = []
                    for value in row_combine_subset:
                        if value in pattern_sum:
                            metrics_hierarchy.append(pattern_sum[value])
                        else:
                            metrics_hierarchy.append("Others")

                    sub_treatment_matrix_metrics = sub_treatment_matrix.assign(Metrics=metrics_hierarchy)
                    

                    metrics_freq = pd.Series(metrics_hierarchy).value_counts()
                    if "Others" in metrics_freq.index:
                        metrics_freq_moveOthers = pd.concat([metrics_freq.drop("Others"), metrics_freq.loc[["Others"]]])
                        highlight_num = len(metrics_freq)-1
                    else:
                        metrics_freq_moveOthers = metrics_freq
                        highlight_num = len(metrics_freq)
                    hierarchy_order = metrics_freq_moveOthers.index.tolist()[::-1]
                    
                    with setting_col:
                        default_color = st.color_picker("Default Color",value="#D9D9D9")
                        hi_color = choose_hi_color(highlight_num)

                    with graph_col:

                        rank_order = treatment_matrix.columns.tolist()
                        var_subset = ["Metrics"] + sub_treatment_matrix.columns.tolist()

                        treatment_order = sub_treatmenet_effect.columns.tolist()[::-1]

                        value_order_subset = {k: treatment_order for k in sub_treatment_matrix.columns.tolist()} | {"Metrics": hierarchy_order}
                        highlight_hierarchy_metrics = hierarchy_order[1:]
                        label_option_simple_treatment = {k: {"fontsize":font_size} for k in rank_order}
                        label_option_metrics_treatment = label_option_simple_treatment | {"Metrics": {"fontsize":font_size}}

                        ax_partial_ordering = plot_hammock(sub_treatment_matrix_metrics, var=var_subset, 
                                                                    value_order=value_order_subset, default_color=default_color,
                                                                    hi_var="Metrics", hi_value=highlight_hierarchy_metrics, 
                                                                    colors=hi_color[::-1],same_scale=var_subset[1:],
                                                                    label_options = label_option_metrics_treatment,
                                                                    width=fig_width, height = fig_height)
                                                                    #min_bar_height_connectors = 0.2)
                        show_hammock(ax_partial_ordering, key="partial_ordering_download")


st.caption("Author: Carmen Alicia Liang Zhen\n"
            "\nReference paper: Liang Zhen C.A., Béliveau A., Schonlau M. (2026+). Visualizing Joint Ranking Distributions in Network Meta-Analysis with Hammock Plots.")

