# Set repo path
repo_path <- "~/Repositories/PMO_MeasuringFitnessperClone/LiveCellImaging/data"
# Set output path
OUT <- "~/Repositories/PMO_MeasuringFitnessperClone/LiveCellImaging/data"

# Load in tracking data; in this case, Ilastik output 
tracking <- read_csv(paste0(repo_path, "/KANU_Timeseries_image_CSV-Table.csv"))

# Name the experiment
exp_name <- "KANU_timeseries"


parent_list <- unique(tracking$parentTrackId)

# Add column indicating the first frame in which a cell is tracked (begins)
# Add column indicating if a cell eventually divides (is_parent) 
tracking_b <- data.frame()
for(i in unique_cells) {
  p <- tracking %>% 
    filter(trackId == i) %>% 
    mutate(begins = min(frame))
  if(i %in% parent_list) {
    p <- p %>% 
      mutate(is_parent = TRUE)
  } else {
    p <- p %>% 
      mutate(is_parent = FALSE)
  }
  tracking_b <- rbind(p, tracking_b)
}

# Prep data for phylogeny creation.
tracking_b <- tracking_b %>% 
  # Remove untracked cells (id == -1)
  filter(trackId != -1) %>% 
  # Set progenitor cells as a original_cell = 0, 
  # all other cells are a original_cell of 1.
  mutate(original_cell = case_when(lineageId == trackId ~0, TRUE ~1)) %>% 
  # Add column counting frames in which a cell is tracked.
  group_by(trackId) %>%
  mutate(length = n()) %>% 
  filter(frame == begins) %>% 
  ungroup()

# Check changes
view(tracking_b[,c(1:5, 63:66)])

# Create the trees; code is largely from collaborator Mark Dane, M.S. @ OHSU Heiser Lab
create_phylo_trees <- function(root_label, tracks_df){
  # Number the nodes
  df <- tracks_df %>%  
    filter(lineageId == root_label) %>%    #####debug by using root 1 only
    arrange(begins) %>%
    group_by(lineageId, is_parent) %>% 
    mutate(node = case_when(!is_parent ~row_number(),
                            TRUE ~0)) %>%
    group_by(lineageId) %>% 
    mutate(node = case_when(original_cell == 0 ~max(node)+1,
                            TRUE ~node)) %>%
    group_by(lineageId, is_parent) %>%
    mutate(node = case_when(is_parent&!original_cell==0 ~max(node)+1:n()-1,
                            TRUE ~node)) %>%
    group_by(lineageId) %>%
    mutate(node = case_when(n()==1 ~1,
                            TRUE ~node))
  df$parent_node <- sapply(df$parentTrackId, function(parentTrackId){
    df$node[df$trackId==parentTrackId]
  })
  
  # Get the number of internal nodes
  Nnode <- sum(df$is_parent)
  
  # Get the length of the root edge
  root.edge <- df$length[df$original_cell == 0]
  
  df <- df %>%
    filter(!original_cell==0)
  
  # Create the phylo edge matrix
  edge <- cbind(as.integer(df$parent_node), as.integer(df$node))
  
  # Use the original cell labels for tip labels
  tip.label <- df %>%
    filter(!is_parent) %>%
    mutate(trackId = as.character(trackId)) %>%
    pull(trackId) %>%
    as.character(df$trackId)
  
  # Add the branch lengths by frames
  edge.length <- df$length
  
  # Build the tree data using the collected information
  tr <- list(edge = edge,
             tip.label = tip.label,
             Nnode = Nnode,
             root.edge = root.edge,
             edge.length = edge.length)
  class(tr) <- "phylo"
  attr(tr, "order") <- "cladwise"
  return(tr)
}

# Get the list of roots which eventually divide (cells in lineage > 1)
roots <- tracking_b %>%
  group_by(lineageId) %>%
  mutate(cells_in_lineage = n()) %>%
  filter(cells_in_lineage >1) %>%
  select(lineageId) %>%
  distinct() %>%
  pull(lineageId)

# Visualize the trees
tree_list <- map(roots, create_phylo_trees, tracks_df = tracking_b)
class(tree_list) <- "multiPhylo"
names(tree_list) <- as.character(roots)
p <- ggplot(tree_list, aes(x, y), multiPhylo = TRUE) +
  geom_tree() +
  xlim(c(0,9)) +
  geom_tiplab(size=3, color="purple") +
  geom_rootedge() +
  theme_tree2() +
  facet_wrap(~.id, ncol=5)

# Be aware that the height and width will need to be adjusted depending on
# how many trees are created and how large these trees are. 
pdf(paste0(OUT, "/phylogenies_", exp_name, ".pdf"), height = 62, width = 12)
print(p)
res <- dev.off()
print(p)
