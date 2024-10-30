# Install required packages if not already installed
if (!require("shiny")) install.packages("shiny")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")

# Load libraries
library(shiny)
library(dplyr)
library(ggplot2)

# Define UI for the application
ui <- fluidPage(
  titlePanel("CSV File Analysis Tool"),
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
      
      h3("Categorical Column Frequency Tables"),
      uiOutput("categorical_tables")
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
  
  # Generate frequency tables for categorical columns
  output$categorical_tables <- renderUI({
    req(data())
    categorical_cols <- sapply(data(), is.factor) | sapply(data(), is.character)
    
    if (any(categorical_cols)) {
      table_output_list <- lapply(names(data())[categorical_cols], function(colname) {
        table_output <- renderPrint({
          table(data()[[colname]])
        })
        tagList(h4(paste("Frequency Table for:", colname)), verbatimTextOutput(paste("table_", colname, sep = "")))
      })
      do.call(tagList, table_output_list)
    } else {
      h4("No categorical columns found for frequency table analysis.")
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)
