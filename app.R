# ============================================
# Bike Sharing Demand Dashboard
# ============================================

library(shiny)
library(shinydashboard)
library(ggplot2)
library(reticulate)
library(dplyr)
library(scales)

# ================================
# 1. Setup & Configuration
# ================================

# Define Primary Color
primary_color <- "#32CD32"

# Use reticulate to source the Python script
# Ensure the path is correct relative to app.R
# We assume app.R is in the root and the script is in "Data product/"
python_script_path <- "Data product/linear-regression.py"

# Check if file exists
if (!file.exists(python_script_path)) {
  stop(paste("Python script not found at:", python_script_path))
}

# Configure reticulate to use the local virtual environment
# We use file.path(getwd(), ".venv") to ensure we point to the local folder
# and not the default ~/.virtualenvs location.
venv_path <- file.path(getwd(), ".venv")
if (dir.exists(venv_path)) {
  use_virtualenv(venv_path, required = TRUE)
}

# Source the Python script
# This will run the top-level code in the script (training the model)
# and make functions like get_predictions available.
source_python(python_script_path)

# Load Data for Tab 1 (Insight)
# Replace with your actual path if different
csv_path <- "Data product/bikehour.csv"
if (file.exists(csv_path)) {
  df <- read.csv(csv_path)
} else {
  # Placeholder if file not found, to prevent crash during dev
  warning("CSV file not found. Using placeholder data.")
  df <- data.frame(
    weekday = rep(0:6, each = 24),
    hr = rep(0:23, 7),
    cnt = sample(10:500, 24 * 7, replace = TRUE),
    weathersit = sample(1:3, 24 * 7, replace = TRUE),
    registered = sample(5:400, 24 * 7, replace = TRUE),
    casual = sample(5:100, 24 * 7, replace = TRUE)
  )
}

# Pre-process data for plotting
# Map integer weekday to labels
weekday_labels <- c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
df$weekday_label <- factor(df$weekday, levels = 0:6, labels = weekday_labels)


# ================================
# 2. UI Definition
# ================================

ui <- dashboardPage(
  skin = "green", # Closest built-in skin, we will override with CSS

  dashboardHeader(
    title = "Bike Sharing Dashboard",
    tags$li(
      class = "dropdown",
      tags$img(src = "logo.png", height = "40px", style = "margin-top: 5px; margin-right: 10px;")
    )
  ),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Insight & Exploration", tabName = "insight", icon = icon("chart-bar")),
      menuItem("Prediction & Decision", tabName = "prediction", icon = icon("robot"))
    )
  ),
  dashboardBody(
    # Custom CSS for Primary Color #32CD32
    tags$head(tags$style(HTML(paste0("
      .skin-green .main-header .logo { background-color: ", primary_color, "; }
      .skin-green .main-header .navbar { background-color: ", primary_color, "; }
      .skin-green .main-sidebar .sidebar .sidebar-menu .active a { border-left-color: ", primary_color, "; }
      .box.box-solid.box-primary>.box-header { background-color: ", primary_color, "; background: ", primary_color, "; }
      .btn-primary { background-color: ", primary_color, "; border-color: ", primary_color, "; }
      .btn-primary:hover { background-color: #28a428; border-color: #28a428; }

      /* Decision Card Styles */
      .decision-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 1.2em;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
      }
      .decision-alert { background-color: #d9534f; } /* Red */
      .decision-surge { background-color: #f0ad4e; } /* Orange */
      .decision-promo { background-color: #5bc0de; } /* Blue */
      .decision-normal { background-color: #5cb85c; } /* Green */
    ")))),
    tabItems(
      # ----------------------------
      # Tab 1: Insight & Exploration
      # ----------------------------
      tabItem(
        tabName = "insight",
        fluidRow(
          box(
            title = "Filters", status = "success", solidHeader = TRUE, width = 3,
            checkboxGroupInput("weather_filter", "Weather Condition:",
              choices = list(
                "Clear/Few Clouds" = 1,
                "Mist/Cloudy" = 2,
                "Light Rain/Snow" = 3,
                "Heavy Rain/Snow" = 4
              ),
              selected = c(1, 2, 3, 4)
            ),
            radioButtons("user_type", "User Type:",
              choices = list(
                "All Users" = "cnt",
                "Membership" = "registered",
                "Non-Membership" = "casual"
              ),
              selected = "cnt"
            ),
            helpText("Note: Registered users are typically commuters, while Casual users are tourists.")
          ),
          box(
            title = "Demand Heatmap: Weekday vs Hour", status = "success", solidHeader = TRUE, width = 9,
            plotOutput("heatmap_plot", height = "500px")
          )
        )
      ),

      # ----------------------------
      # Tab 2: Prediction & Decision
      # ----------------------------
      tabItem(
        tabName = "prediction",
        fluidRow(
          box(
            title = "Scenario Simulation", status = "success", solidHeader = TRUE, width = 3,
            dateInput("pred_date", "Select Date:", value = "2012-07-01"),
            selectInput("pred_weather", "Weather Forecast:",
              choices = list(
                "Clear/Few Clouds" = 1,
                "Mist/Cloudy" = 2,
                "Light Rain/Snow" = 3
              ),
              selected = 1
            ),
            sliderInput("pred_temp", "Temperature (°C):",
              min = -10, max = 40, value = 25
            ),
            actionButton("run_pred", "Run Prediction", icon = icon("play"), class = "btn-primary")
          ),
          box(
            title = "Predicted Demand Curve (24 Hours)", status = "success", solidHeader = TRUE, width = 9,
            plotOutput("pred_plot"),
            uiOutput("decision_card")
          )
        )
      )
    )
  )
)

# ================================
# 3. Server Logic
# ================================

server <- function(input, output) {
  # ----------------------------
  # Tab 1: Insight Logic
  # ----------------------------

  output$heatmap_plot <- renderPlot({
    # Filter data
    filtered_df <- df %>%
      filter(weathersit %in% input$weather_filter)

    # Aggregate data: Mean count by Weekday and Hour
    # We use the selected user type column (cnt, registered, or casual)
    agg_df <- filtered_df %>%
      group_by(weekday_label, hr) %>%
      summarise(avg_demand = mean(.data[[input$user_type]], na.rm = TRUE), .groups = "drop")

    # Plot
    ggplot(agg_df, aes(x = weekday_label, y = hr, fill = avg_demand)) +
      geom_tile(color = "white") +
      scale_fill_gradient(low = "#e5f7e5", high = "#006400", name = "Avg Rentals") +
      scale_y_reverse(breaks = 0:23) + # 0 at top
      labs(x = "Weekday", y = "Hour of Day", title = "Average Bike Rentals Heatmap") +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text = element_text(size = 12),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
      )
  })

  # ----------------------------
  # Tab 2: Prediction Logic
  # ----------------------------

  # Reactive expression to fetch predictions from Python
  predictions <- eventReactive(input$run_pred,
    {
      # Show loading notification
      id <- showNotification("Running Python Model...", duration = NULL, closeButton = FALSE)
      on.exit(removeNotification(id), add = TRUE)

      # Call Python function
      # Ensure inputs are correct types
      date_str <- as.character(input$pred_date)
      weather_cat <- as.integer(input$pred_weather)
      temp_c <- as.numeric(input$pred_temp)

      # Call the function defined in Python script
      res <- get_predictions(date_str, weather_cat, temp_c)

      return(res)
    },
    ignoreNULL = FALSE
  ) # Run once on startup


  output$pred_plot <- renderPlot({
    req(predictions())
    pred_data <- predictions()

    ggplot(pred_data, aes(x = Hour, y = Predicted_Demand)) +
      # Confidence Interval Ribbon
      geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI), fill = primary_color, alpha = 0.2) +
      # Main Line
      geom_line(color = primary_color, size = 1.5) +
      geom_point(color = primary_color, size = 3) +
      # Aesthetics
      scale_x_continuous(breaks = 0:23) +
      labs(
        x = "Hour (0-23)", y = "Predicted Demand",
        title = paste("Demand Prediction for", input$pred_date),
        subtitle = "Shaded area represents 90% Confidence Interval"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 16, face = "bold"),
        axis.text = element_text(size = 12)
      )
  })

  output$decision_card <- renderUI({
    req(predictions())
    pred_data <- predictions()

    # Logic for Manager's Decision
    max_demand <- max(pred_data$Predicted_Demand)
    max_hour <- pred_data$Hour[which.max(pred_data$Predicted_Demand)]

    # System Capacity
    capacity_limit <- 900

    # Logic A: Capacity Breach
    if (max_demand > capacity_limit) {
      dispatch_hour <- max(0, max_hour - 2)
      html_content <- paste0(
        "<div class='decision-card decision-alert'>",
        "<i class='fa fa-exclamation-triangle'></i> Warning: Capacity Breach Imminent!<br>",
        "<small>Predicted peak of ", round(max_demand), " bikes at hour ", max_hour, ".</small><br>",
        "<strong>Action:</strong> Dispatch supply trucks at ", dispatch_hour, ":00 to avoid lost orders.",
        "</div>"
      )
    }
    # Logic B: Pricing Strategy
    else if (max_demand > 700) { # High demand (Top 10% approx)
      html_content <- paste0(
        "<div class='decision-card decision-surge'>",
        "<i class='fa fa-bolt'></i> High Demand Expected<br>",
        "<small>Peak demand: ", round(max_demand), " bikes.</small><br>",
        "<strong>Action:</strong> Enable Surge Pricing to balance supply and maximize revenue.",
        "</div>"
      )
    } else if (max_demand < 200) { # Low demand
      html_content <- paste0(
        "<div class='decision-card decision-promo'>",
        "<i class='fa fa-tags'></i> Low Demand Expected<br>",
        "<small>Peak demand only ", round(max_demand), " bikes.</small><br>",
        "<strong>Action:</strong> Push Coupons/Discounts to stimulate usage.",
        "</div>"
      )
    } else {
      html_content <- paste0(
        "<div class='decision-card decision-normal'>",
        "<i class='fa fa-check-circle'></i> Normal Operations<br>",
        "<small>Demand is within normal range.</small><br>",
        "<strong>Action:</strong> Monitor system status.",
        "</div>"
      )
    }

    HTML(html_content)
  })
}

# ================================
# 4. Run App
# ================================

shinyApp(ui = ui, server = server)
