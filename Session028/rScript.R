# Load required libraries
library(shiny)
library(ggplot2)

# Define UI
ui <- fluidPage(
  titlePanel("Data Visualization with Shiny"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File:", multiple = TRUE, accept = c(".csv")),
      uiOutput("dataset_selector"),
      uiOutput("x_column_selector"),
      uiOutput("y_column_selector"),
      selectInput("plot_type", "Select Plot Type:", 
                  choices = c("Line", "Scatter", "Bar", "Area", 
                              "Histogram", "Boxplot", "Density", "Violin"), 
                  selected = "Line")
    ),
    
    mainPanel(
      textOutput("output_text"),
      plotOutput("data_plot")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive expression to read the uploaded file
  dataset <- reactive({
    req(input$file)  # Ensure a file is uploaded
    read.csv(input$file$datapath[1])  # Read the first uploaded file
  })
  
  # Update dataset selector dynamically
  output$dataset_selector <- renderUI({
    req(input$file)
    selectInput("dataset", "Select Dataset:", choices = input$file$name)
  })
  
  # Update x column selector based on the dataset
  output$x_column_selector <- renderUI({
    req(dataset())
    selectInput("x_column", "Select X Column:", choices = colnames(dataset()))
  })
  
  # Update y column selector based on the dataset
  output$y_column_selector <- renderUI({
    req(dataset())
    selectInput("y_column", "Select Y Column:", choices = colnames(dataset()))
  })
  
  # Render plot based on user selections
  output$data_plot <- renderPlot({
    req(input$x_column, input$y_column)
    df <- dataset()
    
    # Plot based on the selected plot type
    if (input$plot_type == "Line") {
      ggplot(df, aes_string(x = input$x_column, y = input$y_column)) +
        geom_line(color = "blue") +
        labs(title = "Line Plot")
    } else if (input$plot_type == "Scatter") {
      ggplot(df, aes_string(x = input$x_column, y = input$y_column)) +
        geom_point(color = "red") +
        labs(title = "Scatter Plot")
    } else if (input$plot_type == "Bar") {
      ggplot(df, aes_string(x = input$x_column, y = input$y_column)) +
        geom_bar(stat = "identity", fill = "green") +
        labs(title = "Bar Plot")
    } else if (input$plot_type == "Area") {
      ggplot(df, aes_string(x = input$x_column, y = input$y_column)) +
        geom_area(fill = "lightblue") +
        labs(title = "Area Plot")
    } else if (input$plot_type == "Histogram") {
      ggplot(df, aes_string(x = input$x_column)) +
        geom_histogram(binwidth = 1, fill = "purple", color = "white") +
        labs(title = "Histogram")
    } else if (input$plot_type == "Boxplot") {
      ggplot(df, aes_string(x = input$x_column, y = input$y_column)) +
        geom_boxplot(fill = "orange") +
        labs(title = "Boxplot")
    } else if (input$plot_type == "Density") {
      ggplot(df, aes_string(x = input$x_column)) +
        geom_density(fill = "lightgreen") +
        labs(title = "Density Plot")
    } else if (input$plot_type == "Violin") {
      ggplot(df, aes_string(x = input$x_column, y = input$y_column)) +
        geom_violin(fill = "pink") +
        labs(title = "Violin Plot")
    }
  })
  
  # Output text displaying the dataset info
  output$output_text <- renderText({
    req(dataset())
    paste("Data Loaded: ", nrow(dataset()), " rows and ", ncol(dataset()), " columns.")
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
