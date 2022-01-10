library(tidyverse)

create_latex_matrix <- function(mat, prec){
  if (is.matrix(mat)){
    str_vec <- apply(
      mat, 
      1, 
      function(x){str_c(round(x, prec),collapse = ' & ')}
    )
  } else {
    str_vec <- round(mat, prec)
  }
  
  str_vec_cat <- str_c(str_vec, collapse = ' \\\\\n')
  
  cat(str_c(c('\\begin{bmatrix}', str_vec_cat, '\\end{bmatrix}\n'), collapse = ' \n'))
}