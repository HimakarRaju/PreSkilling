# Install required packages if not already installed
if (!require("shiny")) install.packages("shiny")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("caret")) install.packages("caret")
if (!require("stats")) install.packages("stats")

# Load libraries
library(shiny)
library(dplyr)
library(ggplot2)
library(caret)
library(stats)

# Define UI for the application
ui <- fluidPage(
  titlePanel("Dynamic Model Selection based on Uploaded Dataset"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File", accept = ".csv")
    ),
    mainPanel(
      h3("Data Preview"),
      tableOutput("data_head"),
      
      h3("Selected Model"),
      textOutput("model_info"),
      
      h3("Model Output"),
      verbatimTextOutput("model_summary"),
      
      h3("Plot"),
      plotOutput("model_plot")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive expression to load the dataset
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  # Display first few rows of the dataset
  output$data_head <- renderTable({
    req(data())
    head(data())
  })
  
  # Select model based on dataset structure
  output$model_info <- renderText({
    req(data())
    df <- data()
    
    # Check column types
    num_cols <- names(df)[sapply(df, is.numeric)]
    cat_cols <- names(df)[sapply(df, is.factor) | sapply(df, is.character)]
    
    # Determine model type
    if (length(num_cols) >= 2 && length(cat_cols) == 0) {
      "Selected Model: Linear Regression (for numeric predictors and response)"
    } else if (length(num_cols) >= 1 && length(cat_cols) == 1) {
      "Selected Model: Logistic Regression (for binary response)"
    } else if (length(num_cols) >= 2 && nrow(df) > 10) {
      "Selected Model: K-Means Clustering (for multiple numeric columns)"
    } else {
      "Selected Model: Not enough data or appropriate structure for automated model selection."
    }
  })
  
  # Apply selected model
  model_output <- reactive({
    req(data())
    df <- data()
    
    num_cols <- names(df)[sapply(df, is.numeric)]
    cat_cols <- names(df)[sapply(df, is.factor) | sapply(df, is.character)]
    
    if (length(num_cols) >= 2 && length(cat_cols) == 0) {
      # Linear Regression Model
      model <- lm(df[[num_cols[2]]] ~ df[[num_cols[1]]], data = df)
      return(summary(model))
      
    } else if (length(num_cols) >= 1 && length(cat_cols) == 1) {
      # Logistic Regression Model (only works if the categorical column has 2 levels)
      if (nlevels(as.factor(df[[cat_cols[1]]])) == 2) {
        df[[cat_cols[1]]] <- as.factor(df[[cat_cols[1]]])
        model <- glm(df[[cat_cols[1]]] ~ df[[num_cols[1]]], data = df, family = binomial)
        return(summary(model))
      } else {
        return("Categorical variable is not binary for logistic regression.")
      }
      
    } else if (length(num_cols) >= 2 && nrow(df) > 10) {
      # K-Means Clustering
      set.seed(123)  # For reproducibility
      kmeans_model <- kmeans(df[num_cols], centers = 3)
      return(kmeans_model)
      
    } else {
      return("Insufficient or incompatible data for modeling.")
    }
  })
  
  # Render model summary
  output$model_summary <- renderPrint({
    model_output()
  })
  
  # Plot based on selected model
  output$model_plot <- renderPlot({
    req(data())
    df <- data()
    num_cols <- names(df)[sapply(df, is.numeric)]
    cat_cols <- names(df)[sapply(df, is.factor) | sapply(df, is.character)]
    
    if (length(num_cols) >= 2 && length(cat_cols) == 0) {
      # Plot for Linear Regression
      ggplot(df, aes_string(x = num_cols[1], y = num_cols[2])) +
        geom_point(color = "blue") +
        geom_smooth(method = "lm", color = "red") +
        labs(title = "Linear Regression Plot",
             x = num_cols[1],
             y = num_cols[2])
      
    } else if (length(num_cols) >= 1 && length(cat_cols) == 1) {
      # Plot for Logistic Regression (if binary response)
      if (nlevels(as.factor(df[[cat_cols[1]]])) == 2) {
        ggplot(df, aes_string(x = num_cols[1], y = cat_cols[1])) +
          geom_jitter(width = 0.1, height = 0.1, color = "blue") +
          geom_smooth(method = "glm", method.args = list(family = "binomial"), color = "red") +
          labs(title = "Logistic Regression Plot",
               x = num_cols[1],
               y = cat_cols[1])
      }
      
    } else if (length(num_cols) >= 2 && nrow(df) > 10) {
      # Plot for K-Means Clustering
      kmeans_model <- kmeans(df[num_cols], centers = 3)
      df$Cluster <- as.factor(kmeans_model$cluster)
      ggplot(df, aes_string(x = num_cols[1], y = num_cols[2], color = "Cluster")) +
        geom_point(size = 3) +
        labs(title = "K-Means Clustering Plot",
             x = num_cols[1],
             y = num_cols[2])
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)
