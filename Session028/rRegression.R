# Install required packages if not already installed
if (!require("shiny")) install.packages("shiny")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("GGally")) install.packages("GGally")

# Load libraries
library(shiny)
library(dplyr)
library(ggplot2)
library(GGally)

# Define UI for the application
ui <- fluidPage(
  titlePanel("CSV File Auto-Analysis and Pattern Inference"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File", accept = ".csv")
    ),
    mainPanel(
      h3("Data Preview"),
      tableOutput("data_head"),
      
      h3("Summary Statistics"),
      verbatimTextOutput("data_summary"),
      
      h3("Numeric Column Histograms"),
      uiOutput("numeric_histograms"),
      
      h3("Correlation Matrix for Numeric Columns"),
      plotOutput("correlation_matrix"),
      
      h3("Pattern Inference: Regression Analysis"),
      uiOutput("regression_results")
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Reactive expression to load data
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  # Display the first few rows of the data
  output$data_head <- renderTable({
    req(data())
    head(data())
  })
  
  # Display summary statistics for the data
  output$data_summary <- renderPrint({
    req(data())
    summary(data())
  })
  
  # Generate histograms for numeric columns
  output$numeric_histograms <- renderUI({
    req(data())
    numeric_cols <- sapply(data(), is.numeric)
    
    if (any(numeric_cols)) {
      plot_output_list <- lapply(names(data())[numeric_cols], function(colname) {
        plotname <- paste("plot_", colname, sep = "")
        plotOutput(plotname, height = "300px")
      })
      do.call(tagList, plot_output_list)
    } else {
      h4("No numeric columns found for histogram analysis.")
    }
  })
  
  observe({
    req(data())
    numeric_cols <- sapply(data(), is.numeric)
    
    for (colname in names(data())[numeric_cols]) {
      local({
        col <- colname
        output[[paste("plot_", col, sep = "")]] <- renderPlot({
          ggplot(data(), aes_string(x = col)) +
            geom_histogram(bins = 30, fill = "skyblue", color = "black") +
            labs(title = paste("Histogram of", col), x = col, y = "Frequency")
        })
      })
    }
  })
  
  # Generate correlation matrix for numeric columns
  output$correlation_matrix <- renderPlot({
    req(data())
    numeric_data <- data()[, sapply(data(), is.numeric)]
    
    if (ncol(numeric_data) > 1) {
      ggpairs(numeric_data) + theme_minimal() +
        labs(title = "Correlation Matrix of Numeric Columns")
    } else {
      plot.new()
      text(0.5, 0.5, "Not enough numeric columns for correlation analysis.")
    }
  })
  
  # Infer patterns using linear regression analysis
  output$regression_results <- renderUI({
    req(data())
    numeric_cols <- names(data())[sapply(data(), is.numeric)]
    
    if (length(numeric_cols) >= 2) {
      reg_results <- lapply(2:length(numeric_cols), function(i) {
        x_var <- numeric_cols[1]
        y_var <- numeric_cols[i]
        
        model <- lm(data()[[y_var]] ~ data()[[x_var]], data = data())
        summary_text <- summary(model)
        
        plotname <- paste("regression_plot_", y_var, sep = "")
        
        list(
          h4(paste("Regression of", y_var, "on", x_var)),
          renderPrint({ summary_text }),
          plotOutput(plotname)
        )
      })
      
      # Render regression plots
      for (i in 2:length(numeric_cols)) {
        local({
          x_var <- numeric_cols[1]
          y_var <- numeric_cols[i]
          plotname <- paste("regression_plot_", y_var, sep = "")
          
          output[[plotname]] <- renderPlot({
            ggplot(data(), aes_string(x = x_var, y = y_var)) +
              geom_point(color = "blue") +
              geom_smooth(method = "lm", color = "red") +
              labs(title = paste("Regression of", y_var, "on", x_var),
                   x = x_var, y = y_var)
          })
        })
      }
      
      do.call(tagList, unlist(reg_results, recursive = FALSE))
    } else {
      h4("Not enough numeric columns for regression analysis.")
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)
